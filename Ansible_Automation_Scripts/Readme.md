# Steps to use Ansible to automate google cloud instances

## Getting started
You can find the more detailed documentation on https://docs.ansible.com

Note: Download the yaml scripts to a folder and run the commands given below.

## Installation of Ansible and related modules on a linux environment

```bash
sudo apt-get update
sudo apt-get install software-properties-common
sudo apt-add-repository --yes --update ppa:ansible/ansible
sudo apt-get install ansible
```

Make sure that your Google clous shell is intalled on the same computer you are working on.

Update pip and install the requirements

```bash
pip install --upgrade pip
pip install googleauth requests
pip install requests google-auth
```

### Launching instances using Ansible playbooks
```bash
$ ansible-playbook gcp_create_instances.yml -e instances= Instance01,Instance02
```

*-e instances= Instance01,Instance02 - Optional arguement

### Deleting the instances using Ansible playbooks
```bash
$ ansible-playbook gcp_delete_instances.yml -e instances= Instance01,Instance02
```




