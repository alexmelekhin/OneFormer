#!/bin/bash

docker exec --user docker_oneformer -it ${USER}_oneformer \
    /bin/bash -c "cd /home/docker_oneformer; echo ${USER}_oneformer container; echo ; /bin/bash"
