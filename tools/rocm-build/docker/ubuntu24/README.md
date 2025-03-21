## Steps to build the Docker Image

1. Clone this repositry

   ```bash
   git clone -b <release_branch> https://github.com/ROCm/ROCm.git
   ```

2. Go into the OS specific docker directory in build-infra

    ```bash
    cd rocm-build/build/docker/ubuntu24
    ```

3. Build the docker image

    ```bash
    docker build -t <docker Image Name> .
    ```

    replace the `<docker Image Name>` with the new Docker image Name of your choice,

4. After successful build, verify your \<docker Image Name\> in the list all available docker images.

    ```bash
    docker images
    ```
