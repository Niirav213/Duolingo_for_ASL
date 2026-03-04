import base64

image_path = r"C:\Users\ASUS\Downloads\download.jpg"  # your image file name

with open(image_path, "rb") as f:
    encoded = base64.b64encode(f.read()).decode('utf-8')

print(encoded)