# retail_pipeline_fixed.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_timestamp, explode, split, monotonically_increasing_id,
    year, month, dayofmonth, quarter, regexp_replace, trim, coalesce, when, lit
)

from pyspark.sql import functions as F

class RetailPipeline:
    def __init__(self, file_path):
        self.spark = SparkSession.builder.getOrCreate()
        self.file_path = file_path
        self.bronze_df = None
        self.silver_df = None
        self.gold_df = None
        self.customer_dim = None
        self.store_dim = None
        self.product_dim = None
        self.date_dim = None
        self.promotion_dim = None
        self.fact_sales_products = None

    # ------------------------------
    # Bronze Layer
    # ------------------------------
    def load_bronze(self):
        print("Loading dataset into Bronze layer...")
        self.bronze_df = self.spark.read.format("csv") \
            .option("header", "true") \
            .load(self.file_path)
        self.bronze_df.write.format("delta").mode("overwrite").saveAsTable("Bronze_Retail")
        print("✅ Bronze layer saved.")
        self.bronze_df.show(5, truncate=False)

    # ------------------------------
    # Silver layer with robust date parsing + minor casts
    # ------------------------------
    def create_silver(self):
        print("Creating Silver layer (robust date parsing + casting)...")

        # Try several common date/time format patterns, first non-null wins
        parsed_date = coalesce(
            to_timestamp(trim(col("Date")), "M/d/yyyy H:mm"),
            to_timestamp(trim(col("Date")), "M/d/yyyy H:mm:ss"),
            to_timestamp(trim(col("Date")), "yyyy-MM-dd HH:mm:ss"),
            to_timestamp(trim(col("Date")), "M/d/yyyy h:mm a"),
            to_timestamp(trim(col("Date")), "MM/dd/yyyy HH:mm:ss"),
        )

        # Show how many rows fail parsing (so you can inspect)
        invalid_dates_df = self.bronze_df.withColumn("Date_parsed_temp", parsed_date) \
                                         .filter(col("Date_parsed_temp").isNull()) \
                                         .select("Transaction_ID", "Date")
        invalid_count = invalid_dates_df.count()
        if invalid_count > 0:
            print(f"⚠️  Found {invalid_count} rows with unparsed/invalid Date values (showing up to 10):")
            invalid_dates_df.show(10, truncate=False)
        else:
            print("✅ All dates parsed successfully with the attempted formats.")

        # Build silver with parsed Date column and casts
        df = self.bronze_df.withColumn("Date", parsed_date) \
            .withColumn("Discount_Applied", col("Discount_Applied").cast("boolean")) \
            .withColumn("Total_Items", col("Total_Items").cast("int")) \
            .withColumn("Total_Cost", col("Total_Cost").cast("double"))

        # Add MonthNum (for season calc) and compute season when missing
        df = df.withColumn("MonthNum", month(col("Date")))
        season_calc = when((col("MonthNum") == 12) | (col("MonthNum") == 1) | (col("MonthNum") == 2), lit("Winter")) \
            .when((col("MonthNum") >= 3) & (col("MonthNum") <= 5), lit("Spring")) \
            .when((col("MonthNum") >= 6) & (col("MonthNum") <= 8), lit("Summer")) \
            .when((col("MonthNum") >= 9) & (col("MonthNum") <= 11), lit("Fall")) \
            .otherwise(None)

        # If original Season is missing/null, use calculated season; otherwise keep original
        df = df.withColumn("Season", coalesce(col("Season"), season_calc))

        # Drop helper column MonthNum
        df = df.drop("MonthNum")

        self.silver_df = df
        self.silver_df.write.format("delta").mode("overwrite").saveAsTable("Silver_Retail")
        print("✅ Silver layer saved.")
        self.silver_df.show(5, truncate=False)

    # ------------------------------
    # Dimensions (cleaned products + date dim + others)
    # ------------------------------
    def create_dimensions(self):
        print("Creating dimension tables...")

        # Customer Dimension
        self.customer_dim = self.silver_df.select("Customer_Name", "Customer_Category").distinct() \
            .withColumn("Customer_ID", monotonically_increasing_id())
        self.customer_dim.write.format("delta").mode("overwrite").saveAsTable("Customer_Dim")
        print("✅ Customer dimension saved.")
        self.customer_dim.show(5, truncate=False)

        # Store Dimension
        self.store_dim = self.silver_df.select("City", "Store_Type").distinct() \
            .withColumn("Store_ID", monotonically_increasing_id())
        self.store_dim.write.format("delta").mode("overwrite").saveAsTable("Store_Dim")
        print("✅ Store dimension saved.")
        self.store_dim.show(5, truncate=False)

        # Product Dimension (clean product text, explode arrays)
        # 1) strip surrounding [ ] then split on comma
        product_array_col = split(
            regexp_replace(col("Product"), r"^\s*\[|\]\s*$", ""),  # remove leading [ and trailing ]
            r",\s*"  # split on comma + optional spaces
        )

        exploded = self.silver_df.withColumn("Product_Item_Raw", explode(product_array_col))
        # remove surrounding quotes (single or double) and trim whitespace
        exploded = exploded.withColumn(
            "Product_Name",
            trim(regexp_replace(col("Product_Item_Raw"), r"^['\"]|['\"]$", ""))
        ).filter(col("Product_Name").isNotNull() & (trim(col("Product_Name")) != ""))

        self.product_dim = exploded.select("Product_Name").distinct() \
            .withColumn("Product_ID", monotonically_increasing_id())
        self.product_dim.write.format("delta").mode("overwrite").saveAsTable("Product_Dim")
        print("✅ Product dimension saved (clean text, brackets/quotes removed).")
        self.product_dim.show(10, truncate=False)

        # Date Dimension (use parsed Date values)
        date_df = self.silver_df.select(col("Date").alias("Date"), col("Season")).distinct() \
            .filter(col("Date").isNotNull()) \
            .withColumn("Date_ID", monotonically_increasing_id()) \
            .withColumn("Year", year(col("Date"))) \
            .withColumn("Month", month(col("Date"))) \
            .withColumn("Day", dayofmonth(col("Date"))) \
            .withColumn("Quarter", quarter(col("Date")))

        self.date_dim = date_df
        self.date_dim.write.format("delta").mode("overwrite").saveAsTable("Date_Dim")
        print("✅ Date dimension saved.")
        self.date_dim.show(10, truncate=False)

        # Promotion Dimension (clean/rename)
        self.promotion_dim = (
            self.silver_df.select("Promotion").distinct()
            .withColumnRenamed("Promotion", "Promotion_Name")
            .filter(col("Promotion_Name").isNotNull() & (trim(col("Promotion_Name")) != ""))
            .withColumn("Promotion_ID", monotonically_increasing_id())
        )
        self.promotion_dim.write.format("delta").mode("overwrite").saveAsTable("Promotion_Dim")
        print("✅ Promotion dimension saved.")
        self.promotion_dim.show(10, truncate=False)

    # ------------------------------
    # Fact_Sales_Products: uses cleaned Product_Name
    # ------------------------------
    def create_fact_sales_products(self):
        print("Creating Fact_Sales_Products table (star schema with foreign keys)...")

        # Explode products into individual rows
        exploded = self.silver_df.withColumn(
            "Product", F.explode(
                F.split(F.regexp_replace("Product", r"[\[\]']", ""), ",\s*")
            )
        ).withColumn("Product", F.trim(F.col("Product")))

        # === Join with Dimension Tables to replace natural keys with surrogate keys ===
        fact = (
            exploded
            # Join Product
            .join(self.product_dim, exploded["Product"] == self.product_dim["Product_Name"], "left")
            # Join Customer
            .join(self.customer_dim, ["Customer_Name", "Customer_Category"], "left")
            # Join Store
            .join(self.store_dim, ["City", "Store_Type"], "left")
            # Join Promotion
            .join(self.promotion_dim, exploded["Promotion"] == self.promotion_dim["Promotion_Name"], "left")
            # Join Date
            .join(self.date_dim, exploded["Date"] == self.date_dim["Date"], "left")
        )

        # === Select only surrogate keys + measures ===
        fact_clean = fact.select(
            "Transaction_ID",
            "Date_ID",
            "Customer_ID",
            "Store_ID",
            "Product_ID",
            "Promotion_ID",
            "Total_Items",
            "Total_Cost"
        )

        # Save fact table
        fact_clean.write \
            .format("delta") \
            .mode("overwrite") \
            .option("overwriteSchema", "true") \
            .saveAsTable("Fact_Sales_Products")

        self.fact_sales_products = fact_clean

        print("✅ Fact_Sales_Products saved (with proper FKs).")
        self.fact_sales_products.show(10, truncate=False)

    # ------------------------------
    # Gold layer examples (aggregations)
    # ------------------------------
    def create_gold(self):
        print("Creating Gold (aggregations)...")
        self.gold_df = (
            self.silver_df.groupBy("Customer_Name", "City", "Customer_Category")
            .agg({"Total_Cost": "sum", "Total_Items": "sum"})
            .withColumnRenamed("sum(Total_Cost)", "Total_Spent")
            .withColumnRenamed("sum(Total_Items)", "Items_Purchased")
        )
        self.gold_df.write.format("delta").mode("overwrite").saveAsTable("Gold_Retail")
        print("✅ Gold layer saved.")
        self.gold_df.show(5, truncate=False)

    def create_gold_customer(self):
        """Customer-level spend & items."""
        print("Creating Gold (customer spend)...")
        self.gold_customer = (
            self.silver_df.groupBy("Customer_Name", "City", "Customer_Category")
            .agg(F.sum("Total_Cost").alias("Total_Spent"),
                 F.sum("Total_Items").alias("Items_Purchased"))
        )
        self.gold_customer.write.format("delta").mode("overwrite").saveAsTable("Gold_Customer_Spend")
        print("✅ Gold_Customer_Spend saved.")
        self.gold_customer.show(5, truncate=False)

    def create_gold_city_month(self):
        """Sales aggregated by city & month."""
        print("Creating Gold (sales by city & month)...")
        self.gold_city_month = (
            self.silver_df.groupBy(
                F.col("City"),
                F.year("Date").alias("Year"),
                F.month("Date").alias("Month")
            )
            .agg(F.sum("Total_Cost").alias("Total_Sales"),
                 F.sum("Total_Items").alias("Total_Items"))
        )
        self.gold_city_month.write.format("delta").mode("overwrite").saveAsTable("Gold_Sales_By_City_Month")
        print("✅ Gold_Sales_By_City_Month saved.")
        self.gold_city_month.show(5, truncate=False)

    def create_gold_promotion(self):
        """Promotion effectiveness analysis."""
        print("Creating Gold (promotion effectiveness)...")
        self.gold_promotion = (
            self.silver_df.groupBy("Promotion")
            .agg(F.countDistinct("Transaction_ID").alias("Num_Transactions"),
                 F.sum("Total_Cost").alias("Revenue"))
        )
        self.gold_promotion.write.format("delta").mode("overwrite").saveAsTable("Gold_Promotion_Effectiveness")
        print("✅ Gold_Promotion_Effectiveness saved.")
        self.gold_promotion.show(5, truncate=False)

    def create_gold_customer_ltv(self):
        """Customer lifetime value."""
        print("Creating Gold (customer lifetime value)...")
        self.gold_ltv = (
            self.silver_df.groupBy("Customer_Name")
            .agg(F.sum("Total_Cost").alias("Lifetime_Spend"),
                 F.countDistinct("Transaction_ID").alias("Num_Transactions"))
        )
        self.gold_ltv.write.format("delta").mode("overwrite").saveAsTable("Gold_Customer_LTV")
        print("✅ Gold_Customer_LTV saved.")
        self.gold_ltv.show(5, truncate=False)

    def create_gold_basket(self):
        """Basket analysis (co-purchased products)."""
        print("Creating Gold (basket analysis)...")

        basket = (
            self.fact_sales_products.alias("a")
            .join(
                self.fact_sales_products.alias("b"),
                (F.col("a.Transaction_ID") == F.col("b.Transaction_ID")) &
                (F.col("a.Product_ID") < F.col("b.Product_ID"))
            )
            .select(
                F.col("a.Product_ID").alias("Product_A"),
                F.col("b.Product_ID").alias("Product_B")
            )
            .groupBy("Product_A", "Product_B")
            .count()
            .orderBy(F.desc("count"))
        )

        self.gold_basket = basket
        basket.write.format("delta").mode("overwrite").saveAsTable("Gold_Basket_Analysis")
        print("✅ Gold_Basket_Analysis saved.")
        basket.show(10, truncate=False)
        

    # ============================
    # RUN ALL GOLD TOGETHER
    # ============================
    def run_gold_pipeline(self):
        print("\n🚀 Running Gold Pipeline (all marts)...")
        self.create_gold_customer()
        self.create_gold_city_month()
        self.create_gold_promotion()
        self.create_gold_customer_ltv()
        self.create_gold_basket()
        print("🎉 All Gold marts created successfully!")

    # ------------------------------
    # Verify counts and quick checks
    # ------------------------------
    def verify_counts(self):
        print("Verifying record counts...")
        print(f"Bronze count: {self.spark.table('Bronze_Retail').count()}")
        print(f"Silver count: {self.spark.table('Silver_Retail').count()}")
        print(f"Gold count: {self.spark.table('Gold_Retail').count()}")
        print(f"Customer_Dim count: {self.spark.table('Customer_Dim').count()}")
        print(f"Store_Dim count: {self.spark.table('Store_Dim').count()}")
        print(f"Product_Dim count: {self.spark.table('Product_Dim').count()}")
        print(f"Date_Dim count: {self.spark.table('Date_Dim').count()}")
        print(f"Promotion_Dim count: {self.spark.table('Promotion_Dim').count()}")
        print(f"Fact_Sales_Products count: {self.spark.table('Fact_Sales_Products').count()}")

# ------------------------------
# Run pipeline
# ------------------------------
if __name__ == "__main__":
    file_path = "./Data/Retail_Transactions_Dataset.csv"
    pipeline = RetailPipeline(file_path)
    pipeline.load_bronze()
    pipeline.create_silver()
    pipeline.create_dimensions()
    pipeline.create_fact_sales_products()
    pipeline.run_gold_pipeline()
    pipeline.verify_counts()
