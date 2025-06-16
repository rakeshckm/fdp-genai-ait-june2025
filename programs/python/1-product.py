class Product:
    def __init__(self, code, name, supplier, price):
        self.code = code
        self.name = name
        self.supplier = supplier
        self.price = price

    def info(self):
        print(f"Code: {self.code}")
        print(f"Name: {self.name}")
        print(f"Supplier: {self.supplier}")
        print(f"Price: {self.price}")
    def __str__(self):
        return (f"Code: {self.code}\n"
                f"Name: {self.name}\n"
                f"Supplier: {self.supplier}\n"
                f"Price: {self.price}")


# Example usage:
product1 = Product("P001", "Laptop", "ABC Corp", 50000)
product1.info()
print("product 1", product1)

product2 = Product("P002", "Mobile", "Samsung", 60000)
product2.info()

class ProductManagement:
    def __init__(self):
        self.products = []

    def addProduct(self, product):
        self.products.append(product)

    def listProduct(self):
        print("Listing all products:")
        for product in self.products:
            product.info()
            print("-" * 20)

    def searchbyname(self, name):
        print(f"Searching for product with name: {name}")
        found = False
        for product in self.products:
            if product.name.lower() == name.lower():
                product.info()
                found = True
        if not found:
            print(f"No product found with name: {name}")

    def findbypricerange(self, min_price, max_price):
        found = False
        for product in self.products:
            if min_price <= product.price <= max_price:
                product.info()
                print("-" * 20)
                found = True
        if not found:
            print(f"No products found in price range {min_price} to {max_price}")

# Example usage:
pm = ProductManagement()
p1 = Product("P001", "Laptop", "ABC Corp", 50000)
pm.addProduct(p1)
p2 = Product("P002", "Mobile", "Samsung", 80000)
pm.addProduct(p2)
p3 = Product("P003", "earphone", "Boat", 2000)
pm.addProduct(p3)

pm.listProduct()
pm.searchbyname("Mobile")
pm.searchbyname("Tablet")  # This should print "No product found with name: Tablet"
pm.findbypricerange(1000, 60000)  # Should list products within this price range
pm.findbypricerange(1000, 2000)  # Should list products within this price range

