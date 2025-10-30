import requests

url = "http://127.0.0.1:8000/photo/enhance"
file_path = r"C:\Users\assas\Pictures\low.jpg"  # Input image
output_path = r"C:\Users\assas\Pictures\enhanced_low.jpg"  # Output image

try:
    with open(file_path, "rb") as f:
        files = {"file": (file_path, f, "image/jpeg")}
        response = requests.post(url, files=files)

    if response.status_code == 200:
        with open(output_path, "wb") as out:
            out.write(response.content)
        print(f"✅ Enhanced image saved as {output_path}")
    else:
        print(f"❌ Request failed with status code {response.status_code}")
        try:
            print("Server response:", response.json())
        except Exception:
            print("Server response (raw text):", response.text)

except Exception as e:
    print(f"❌ Error during request: {e}")
