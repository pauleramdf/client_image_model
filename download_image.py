import requests


def download_image(url, save_path):
    try:
        # Send a GET request to the URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Write the image data to a file in binary mode
            with open(save_path, 'wb') as file:
                file.write(response.content)
            print(f"Image successfully downloaded: {save_path}")
        else:
            print(f"Failed to download image. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")


# URL of the image to be downloaded
image_url = 'https://imagens.f5news.com.br/arquivos/2021/03/aracaju_antiga_808581615983881.jpg'

# Path where the image will be saved
save_path = 'sample_image.jpg'

# Download the image
download_image(image_url, save_path)