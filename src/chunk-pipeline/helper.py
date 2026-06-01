import subprocess

def get_doc_repo_git_info(repo_path: str) -> tuple[str, str]:
    """
    Extracts the latest commit date (YYYYMMDD) and short hash from the target docs repo.
    This is used in naming of chunks file, to keep record of chunks based on commits on repo.
    """
    try:
        # Get commit date in YYYYMMDD format
        date_cmd = ["git", "-C", repo_path, "log", "-1", "--format=%cd", "--date=format:%Y%m%d"]
        commit_date = subprocess.check_output(date_cmd).decode("utf-8").strip()
        
        # Get short commit hash
        hash_cmd = ["git", "-C", repo_path, "rev-parse", "--short", "HEAD"]
        commit_hash = subprocess.check_output(hash_cmd).decode("utf-8").strip()
        
        return commit_date, commit_hash
    except Exception as e:
        # Fallback if git command fails or directory isn't a git repo
        import datetime
        current_date = datetime.datetime.now().strftime("%Y%m%d")
        return current_date, "unknown"