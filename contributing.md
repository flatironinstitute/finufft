This repository is formatted according to the .clang-format in the root directory. 
Please enable the reformatting hook before committing your changes.
See [pre-commit](https://pre-commit.com/) for more information.
A quick summary:
```
pip install pre-commit
pre-commit install
```

We also suggest to configure your IDE to use the same formatting settings.

Another suggestion is to ignore the formatting commits in your git configuration:
```
git config blame.ignoreRevsFile .git-blame-ignore-revs
```
