
import subprocess

def clone(remote, repo):
    subprocess.run(['git', 'clone', '--recurse-submodules', remote, str(repo)], check=True)

def checkout(repo, commit):
    subprocess.run(['git', 'checkout', str(commit)], cwd=str(repo), check=True)

def is_dirty(repo):
    p = subprocess.run(['git', 'diff-index', '--quiet', 'HEAD'], check=False, cwd=str(repo))
    return p.returncode != 0
