# Apacheworkflow







airflow-docker/
├── dags/                 # Local folder for synced DAGs
├── utils                   # Example DAG
├── .env                  # Environment variables (e.g., AIRFLOW_UID)
├── .gitignore            # Git ignore file
├── docker-compose.yml    # Docker Compose configuration
├── README.md             # This file
└── .ssh/                 # (Optional) SSH key for Git access
    └── id_ed25519


## 1. Setting Up the Virtual Environment

### Create a Virtual Environment

Run the following command to create a virtual environment:

```sh
python -m venv venv
```

### Activate the Virtual Environment

#### Linux and macOS
```sh
source venv/bin/activate
```

#### Windows
```sh
venv\Scripts\activate
```

### Install Dependencies

Ensure you have the necessary dependencies installed:

```sh
python -m pip install --upgrade pip
pip install -r requirements.txt
```

start docker
```sh
docker-compose up -d
```
stop docker
```sh
docker-compose down
```

access airflow
```sh
http://localhost:8080
```
