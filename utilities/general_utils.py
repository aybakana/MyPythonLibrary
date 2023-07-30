import platform
import time

def get_os_info():
    """
    Get information about the operating system.

    Returns:
        dict: A dictionary containing information about the operating system.
              The dictionary includes keys: 'name', 'version', 'architecture'.
    """
    os_name = platform.system()
    os_version = platform.release()
    os_architecture = platform.machine()

    os_info = {
        'name': os_name,
        'version': os_version,
        'architecture': os_architecture
    }

    return os_info



def get_current_timestamp():
    """
    Get the current timestamp as a string.

    Returns:
        str: Current timestamp in the format 'YYYY-MM-DD HH:MM:SS'.
    """
    return time.strftime('%Y-%m-%d %H:%M:%S')

def check_internet_connection():
    """
    Check if the device has an active internet connection.

    Returns:
        bool: True if the device has an internet connection, False otherwise.
    """
    try:
        # You can replace 'www.google.com' with any other reliable website
        # to check internet connectivity.
        import requests
        response = requests.get('https://www.google.com', timeout=5)
        return response.status_code == 200
    except requests.ConnectionError:
        return False

def generate_random_string(length):
    """
    Generate a random string of a specified length.

    Args:
        length (int): The length of the random string to be generated.

    Returns:
        str: Random string of the specified length.
    """
    import random
    import string
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

def convert_bytes_to_mb(bytes):
    """
    Convert bytes to megabytes.

    Args:
        bytes (int): Size in bytes.

    Returns:
        float: Size in megabytes (MB).
    """
    return bytes / (1024 ** 2)

def mount_drive():
    from google.colab import drive
    drive.mount('/content/drive')