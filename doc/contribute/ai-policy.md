<!-- markdownlint-disable MD013 -->

# AI Usage Policy

**Note:** Some Xarray developers use AI tools as part of our development workflow.
We assume this is now common. Tools, patterns, and norms are evolving fast — this
policy aims to avoid restricting contributors' choice of tooling while ensuring that:

- Reviewers are not overburdened
- Contributions can be maintained
- The submitter can vouch for and explain all changes

To that end this policy applies regardless of whether the code was written by hand, with
AI assistance, or generated entirely by an AI tool.

## Core Principle: Changes

If you submit a pull request, you are responsible for understanding and having fully reviewed
the changes. You must be able to explain why each change is correct[^1] and how it fits into
the project. Strive to minimize changes to ease the burden on reviewers — avoid
including unnecessary or loosely related changes.

[^1]:
    You may also open a draft PR with changes in order to discuss and receive feedback on the
    best approach if you are not sure what the best way forward is.

## Core Principle: Communication

PR descriptions, issue comments, and review responses must be your own words. The
substance and reasoning must come from you. Do not paste AI-generated text as
comments or review responses. Please attempt to be concise.

Using AI to improve the language of your writing (grammar, phrasing, spelling, etc.) is
acceptable. Be careful that it does not introduce inaccurate details in the process.

## Code and Tests

### Review Every Line

You must have personally reviewed and understood all changes before submitting.

If you used AI to generate code, you are expected to have read it critically and
tested it. As with a hand-written PR, the description should explain the approach
and reasoning behind the changes. Do not leave it to reviewers to figure out what
the code does and why.

#### Not Acceptable

> I pointed an agent at the issue and here are the changes

> This is what Claude came up with. 🤷

#### Acceptable

> I iterated multiple times with an agent to produce this. The agent wrote the code at my direction,
> and I have fully read and validated the changes.

> I pointed an agent at the issue and it generated a first draft. I reviewed the changes thoroughly and understand the implementation well.

### Large AI-Assisted Contributions

Generating code with agents is fast and easy. Reviewing it is not. Making a PR with a large diff
shifts the burden from the contributor to the reviewer. To guard against this asymmetry:

If you are planning a large AI-assisted contribution (e.g., a significant refactor, a
framework migration, or a new subsystem), **open an issue first** to discuss the scope
and approach with maintainers. This helps us decide if the change is worthwhile, how
it should be structured, and any other important decisions.

Maintainers reserve the right to close PRs where the scope makes meaningful review
impractical, or when they suspect this policy has been violated. Similarly they may request
that large changes be broken into smaller, reviewable pieces.

## Documentation

The same core principles apply to both code and documentation You must review the result
for accuracy and are ultimately responsible for all changes made. Xarray has domain-specific
semantics that AI tools frequently get wrong. Do not submit documentation that you
haven't carefully read and verified.
