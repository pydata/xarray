import re
import textwrap

import git
from tlz.itertoolz import last, unique

co_author_re = re.compile(r"Co-authored-by: (?P<name>[^<]+?) <(?P<email>.+)>")


def main():
    repo = git.Repo(".")

    most_recent_release = last(repo.tags)

    # extract information from commits
    contributors = {}
    for commit in repo.iter_commits(f"{most_recent_release.name}.."):
        matches = co_author_re.findall(commit.message)
        if matches:
            contributors.update({email: name for name, email in matches})
        contributors[commit.author.email] = commit.author.name

    # deduplicate and ignore
    # TODO: extract ignores from .github/release.yml
    ignored = ["dependabot", "pre-commit-ci"]
    unique_contributors = unique(
        contributor
        for contributor in contributors.values()
        if contributor.removesuffix("[bot]") not in ignored
    )

    sorted_ = sorted(unique_contributors)
    if len(sorted_) > 1:
        names = f"{', '.join(sorted_[:-1])} and {sorted_[-1]}"
    else:
        names = "".join(sorted_)

    statement = textwrap.dedent(
        f"""\
    Thanks to the {len(sorted_)} contributors to this release:
    {names}
    """.rstrip()
    )

    print(statement)


if __name__ == "__main__":
    main()
