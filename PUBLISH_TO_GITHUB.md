# Publish to GitHub

Run these commands after creating an empty GitHub repository named `ZSH` under the `sagarkorde` account.

```bash
cd C:\Users\sagar\Desktop\ZSH
git add .
git commit -m "Initial paper submission release"
git push -u origin main
git tag v1.0-paper-submission
git push origin v1.0-paper-submission
```

After publishing:

1. replace the placeholder URL in the manuscript with `https://github.com/sagarkorde/ZSH`
2. update `CITATION.cff` with the final repository URL
3. create a GitHub release from tag `v1.0-paper-submission`
