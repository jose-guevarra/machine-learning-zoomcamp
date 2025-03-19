FROM public.ecr.aws/amazonlinux/amazonlinux:latest


RUN yum update -y \
    && yum install -y sudo vim python3-pip git python3.11 python3.11-pip glibc-langpack-en


# Use venv dir local to Pipfile
ENV PIPENV_VENV_IN_PROJECT=1

RUN pip3.11 install pipenv


# create remote user to wheel group, then allow passwordless sudo for wheel group
#RUN /usr/sbin/adduser -rm -d /workspaces -s /bin/bash -g wheel -u 1001 $REMOTE_USER \
#    && echo '%wheel ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

#USER $REMOTE_USER
