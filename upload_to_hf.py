from huggingface_hub import HfApi, create_repo

repo_id = "Nesmaaaa/image-caption-generator"

create_repo(repo_id, repo_type="model", private=False, exist_ok=True)

api = HfApi()

files_to_upload = {
    "checkpoints/best_model.pt": "best_model.pt",
    "checkpoints/vocab.pkl": "vocab.pkl",
    "checkpoints/confg.json": "config.json",
}

for local_path, repo_path in files_to_upload.items():
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id=repo_id,
    )
    print(f"✅ uploaded: {repo_path}")