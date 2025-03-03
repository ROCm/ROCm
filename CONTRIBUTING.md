<head>
  <meta charset="UTF-8">
  <meta name="description" content="Contributing to ROCm">
  <meta name="keywords" content="ROCm, contributing, contribute, maintainer, contributor">
</head>

# Contribute to ROCm

AMD values and encourages contributions to our code and documentation. If you want to contribute
to our ROCm repositories, first review the following guidance. For documentation-specific information,
see [Contributing to ROCm docs](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

ROCm is a software stack made up of a collection of drivers, development tools, and APIs that enable
GPU programming from low-level kernel to end-user applications. Because some of our components
are inherited from external projects (such as
[LLVM](https://github.com/ROCm/llvm-project) and
[Kernel driver](https://github.com/ROCm/ROCK-Kernel-Driver)), these use
project-specific contribution guidelines and workflow. Refer to their repositories for more information.
All other ROCm components follow the workflow described in the following sections.

## Development workflow

ROCm uses GitHub to host code, collaborate, and manage version control. We use pull requests (PRs)
for all changes within our repositories. We use
[GitHub issues](https://github.com/ROCm/ROCm/issues) to track known issues, such as
bugs.

### Issue tracking

Before filing a new issue, search the
[existing issues](https://github.com/ROCm/ROCm/issues) to make sure your issue isn't
already listed.

General issue guidelines:

* Use your best judgement for issue creation. If your issue is already listed, upvote the issue and
  comment or post to provide additional details, such as how you reproduced this issue.
* If you're not sure if your issue is the same, err on the side of caution and file your issue.
  You can add a comment to include the issue number (and link) for the similar issue. If we evaluate
  your issue as being the same as the existing issue, we'll close the duplicate.
* If your issue doesn't exist, use the issue template to file a new issue.
  * When filing an issue, be sure to provide as much information as possible, including script output so
    we can collect information about your configuration. This helps reduce the time required to
    reproduce your issue.
  * Check your issue regularly, as we may require additional information to successfully reproduce the
    issue.

### Pull requests

When you create a pull request, you should target the default branch.  Our repositories typically use the **develop** branch as the default integration branch.

When creating a PR, use the following process. Note that each repository may include additional,
project-specific steps. Refer to each repository's PR process for any additional steps.

* Identify the issue you want to fix
* Target the default branch (usually the **develop** branch) for integration
* Ensure your code builds successfully
* Each component has a suite of test cases to run; include the log of the successful test run in your PR
* Do not break existing test cases
* New functionality is only merged with new unit tests
  * If your PR includes a new feature, you must provide an application or test so we can ensure that the
    feature works and continues to be valid in the future
* Tests must have good code coverage
* Submit your PR and work with the reviewer or maintainer to get your PR approved
* Once approved, the PR is brought onto internal CI systems and may be merged into the component
  during our release cycle, as coordinated by the maintainer
* We'll inform you once your change is committed

By creating a PR, you agree to allow your contribution to be licensed under the
terms of the LICENSE.txt file in the corresponding repository. Different repositories may use different
licenses. [!IMPORTANT]

You can look up each license on the [ROCm licensing](https://rocm.docs.amd.com/en/latest/about/license.html) page.

### New feature development

Use the [GitHub Discussion forum](https://github.com/ROCm/ROCm/discussions)
(Ideas category) to propose new features. Our maintainers are happy to provide direction and
feedback on feature development.

### Documentation

Submit ROCm documentation changes to our
[documentation repository](https://github.com/ROCm/ROCm). You must update
documentation related to any new feature or API contribution.

Note that each ROCm project uses its own repository for documentation.

## Future development workflow

The current ROCm development workflow is GitHub-based. If, in the future, we change this platform,
the tools and links may change. In this instance, we will update contribution guidelines accordingly.
