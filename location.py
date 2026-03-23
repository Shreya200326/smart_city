import geocoder
import requests
# Step 1: Use your exact coordinates (Decimal Degrees)
lat = 28.507939478229538
lon = 77.52096828808291
# Step 2: Reverse geocode using Geoapify
API_KEY = "cf1440e427d842d9bbdb28381f12e071"  # replace with your Geoapify API key
url = f"https://api.geoapify.com/v1/geocode/reverse?lat={lat}&lon={lon}&apiKey={API_KEY}"
response = requests.get(url)
data = response.json()
# Step 3: Get the address
if data.get("features"):
    address = data["features"][0]["properties"]["formatted"]
else:
    address = "Address not found"
# Step 4: Print results
print(f"Latitude: {lat}, Longitude: {lon}")
print(f"Address: {address}")
