import requests
from bs4 import BeautifulSoup
import pandas as pd

class ECommerceScraper:
    def __init__(self, base_url):
        self.base_url = base_url

    def scrape_product_data(self, num_pages):
        all_products = []
        for page in range(1, num_pages + 1):
            url = f"{self.base_url}/products?page={page}"
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            products = soup.find_all('div', class_='product-item')
            for product in products:
                name = product.find('h2', class_='product-name').text.strip()
                price = float(product.find('span', class_='price').text.strip().replace('$', ''))
                rating = float(product.find('div', class_='rating').text.strip().split('/')[0])
                all_products.append({'name': name, 'price': price, 'rating': rating})

        return pd.DataFrame(all_products)

    def save_to_csv(self, df, filename):
        df.to_csv(f"data/raw/{filename}", index=False)

if __name__ == "__main__":
    scraper = ECommerceScraper("https://example-ecommerce.com")
    product_data = scraper.scrape_product_data(num_pages=10)
    scraper.save_to_csv(product_data, "product_data.csv")