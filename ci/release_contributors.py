import re
import textwrap

import git
from tlz.itertoolz import last, unique

co_author_re = re.compile(r"Co-authored-by: (?P<name>[^<]+?) <(?P<email>.+)>")


ignored = [
    {"name": "dependabot[bot]"},
    {"name": "pre-commit-ci[bot]"},
    {
        "name": "Claude",
        "email": [
            "noreply@anthropic.com",
            "claude@anthropic.com",
            "no-reply@anthropic.com",
        ],
    },
]


def is_ignored(name, email):
    # linear search, for now
    for ignore in ignored:
        if ignore["name"] != name:
            continue
        ignored_email = ignore.get("email")
        if ignored_email is None or email in ignored_email:
            return True

    return False


def main():
    repo = git.Repo(".")

    most_recent_release = last(list(repo.tags))

    # extract information from commits
    contributors = {}
    for commit in repo.iter_commits(f"{most_recent_release.name}.."):
        matches = co_author_re.findall(commit.message)
        if matches:
            contributors.update({email: name for name, email in matches})
        contributors[commit.author.email] = commit.author.name

    # deduplicate and ignore
    # TODO: extract ignores from .github/release.yml
    unique_contributors = unique(
        name for email, name in contributors.items() if not is_ignored(name, email)
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
