
from dotenv import dotenv_values

values = dotenv_values(".env")

with open(".env.docker", "w", encoding="utf-8") as file:
    for key, value in values.items():
        if value is not None:
            escaped = value.replace("\n", "\\n")
            file.write(f"{key}={escaped}\n")

print("Created .env.docker successfully")
