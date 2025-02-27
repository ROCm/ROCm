## Steps to build the ROCm Docker image

1. Clone this repository.

   ```bash
   git clone -b <release_branch> https://github.com/ROCm/ROCm.git
   ```

2. Navigate to the Ubuntu 22 Docker directory.

    ```bash
    cd ROCm/tools/rocm-build/docker/ubuntu22
    ```

3. Build the Docker image.

    ```bash
   docker build -t rocm-ubuntu22
    ```

>[!TIP]
>You can replace rocm-ubuntu22 with any name you prefer for your Docker image.

4. Verify the image was created successfully.

    ```bash
    docker images
    ```

>[!TIP]
>Look for your image name in the list of available Docker images. You should see your newly created image with the name specified in step 3.
