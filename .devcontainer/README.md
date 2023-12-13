# Usage instructions

Note that both options will require you to install docker. Please see the appropriate instructions for your platform.

## Option 1: Let VSCode build the image

If you don't want to install or mess with the devcontainer CLI, copy the devcontainer json file for the Fortran compiler you would like to test to the file `devcontainer.json` and then in VSCode just use "Dev Containers: Open Workspace in Container..."

## Option 2: Build the image yourself

By building the image yourself, you can have it readily available so that "Open Workspace in Container" launches more quickly

This will require you to install the devcontainer CLI, which should be as simply as hitting Ctrl+Shift+P in VSCode and selecting "Install devcontainer CLI"

With the CLI installed, from your terminal run the following (changing `aflang` as appropriate, just make sure to change it in both places)

`devcontainer build --config=devcontainer.aflang.json --image-name=primadevcontainer:aflang`

Then create a `devcontainer.json` in this folder containing `{ "image": "primadevcontainer:aflang" }` and it should be ready for use.