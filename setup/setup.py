import os
import socket
import sys
import virtualenv

# Setup information.
workspace_name = 'workspace'
env_name = 'env'
requirements = 'setup/requirements.txt'

# Data location mapping.
datastore = {
        'default':{
            'datasets':{
                },
            'visdom':{
                },
            }
        }

def setup_workspace():
    """Sets up the workspace for the project by creating a virtual environment,
    installing the required packages and creating symlinks for the dataset.
    """

    hostname = socket.gethostname()

    # Create the workspace folder if it does not exist.
    workspace_path = os.path.abspath(workspace_name)
    if not os.path.isdir(workspace_path):
        print('Creating workspace: {}'.format(workspace_path))
        os.makedirs(workspace_path)

    # Create the virtual environment if it does not exist.
    env_path = os.path.join(workspace_path, env_name)
    if not os.path.isdir(env_path):
        print('Creating environment: {}'.format(env_path))
        virtualenv.create_environment(home_dir=env_path, site_packages=False)

    # Activate the environment.
    print('Activating virtual environment.')
    activate_script = os.path.join(env_path, 'bin', 'activate_this.py')
    execfile(activate_script, dict(__file__=activate_script))
    if hasattr(sys, 'real_prefix'):
        print('Activation done.')
    else:
        print('Activation failed!')

    # Install the requirements.
    # Since using the pip API is not working, we are just running bash commands
    # through Python's os library.
    commands = [
            'python -m pip install --upgrade pip',
            'pip install --upgrade -r {}'.format(requirements)
            ]
    print('Installing requirements.')
    for command in commands:
        os.system(command)
    print('Installed requirements.')

    # Set up symlinks and project folders.
    from termcolor import colored
    print('Setting up for {}'.format(colored(hostname, 'red')))

    paths = None
    if hostname not in datastore:
        print("""Hostname {} is not setup. Using default Cluster""".format(colored(hostname, 'red')))
        paths = datastore['default']
    else:
        paths = datastore[hostname]

    for parent, files in paths.items():
        print('Preparing {}'.format(colored(parent)))
        parent_path = os.path.join(workspace_path, parent)
        if not os.path.isdir(parent_path):
            os.makedirs(parent_path)

        for key, value in files.items():
            target_path = os.path.join(parent_path, key)
            print('\t{} -> {}'.format(colored(target_path, 'green'),
                colored(value, 'blue')))
            if not os.path.islink(target_path):
                os.system('ln -s "{}" "{}"'.format(value, target_path))
    print('Setting up symlinks finished.')

    print("""Setup finished.\nRun "source {}/bin/activate" """.format(env_path))

if __name__ == '__main__':
    setup_workspace()