import typer
from commands import preprocess, train, evaluate, visualize

app = typer.Typer()

app.command()(preprocess.main)
app.command()(train.main)
app.command()(evaluate.main)
app.command()(visualize.main)

if __name__ == "__main__":
	app()
