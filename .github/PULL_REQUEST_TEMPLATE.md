<!-- Feel free to remove check-list items aren't relevant to your change -->

- [ ] Closes #xxxx
- [ ] Tests added
- [ ] Passes `pre-commit run --all-files`
- [ ] User visible changes (including notable bug fixes) are documented in `whats-new.rst`
- [ ] New functions/methods are listed in `api.rst`


<sub>
<h3>
  Overriding CI behaviors
</h3>
   By default, the upstream dev CI is disabled on pull request and push events. You can override this behavior per commit by adding a <tt>[test-upstream]</tt> tag to the first line of the commit message. For documentation-only commits, you can skip the CI per commit by adding a <tt>[skip-ci]</tt> tag to the first line of the commit message
</sub>
