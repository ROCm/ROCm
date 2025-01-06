# ROCm-CI Azure DevOps Pipelines

ROCm-CI Azure DevOps Pipelines contains markup language files that orchestrate build and test pipelines for ROCm components using [Azure DevOps](https://dev.azure.com/ROCm-CI/ROCm-CI/_build).

## Project Organization

- `/.azuredevops/variables-global.yml` - set of global variables accessible across any and all pipelines
  - protected keywords such as tokens and passwords are kept as secrets within the Azure DevOps project
- `/.azuredevops/components` - the sequence of templated steps for the job that checks out source, builds, packages, and runs tests for a ROCm repo
- `/.azuredevops/scheduled` - the sequence of templated steps for jobs that are schedule-based and not tied to a specific ROCm repo
- `/.azuredevops/tag-builds` - yml files to orchestrate manual builds based on specific tags (e.g., releases) without needing the corresponding yaml file in the component's repo
- `/.azuredevops/templates` - reusable yml files representing the templated steps that form the sequences in the above directories

### Per ROCm Repo

- `/.azuredevops/rocm-ci.yml` - contains the CI and PR trigger definitions associated with that repo, pointing to the corresponding yml file in the components folder in this central repository

## Key Azure Reference Links

- [Pipeline Basics](https://learn.microsoft.com/en-us/azure/devops/pipelines/get-started/key-pipelines-concepts?view=azure-devops)
- [Templates](https://learn.microsoft.com/en-us/azure/devops/pipelines/process/templates?view=azure-devops&pivots=templates-includes)
- [Use Predefined Variables](https://learn.microsoft.com/en-us/azure/devops/pipelines/build/variables?view=azure-devops&tabs=yaml)
- [YAML schema](https://learn.microsoft.com/en-us/azure/devops/pipelines/yaml-schema/?view=azure-pipelines&viewFallbackFrom=azure-devops)
- [Azure Pipelines Task Index](https://learn.microsoft.com/en-us/azure/devops/pipelines/tasks/reference/?view=azure-pipelines)

## Disclaimer

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard versionchanges, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated.AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.
