import click

@click.group()
def main():
    pass

@main.command()
def histo():
    pass

@main.command()
def select_landmarks():
    pass

@main.command()
def fit_landmarks():
    pass

@main.command()
def project():
    pass


