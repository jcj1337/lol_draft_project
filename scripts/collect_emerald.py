"""
Emerald version of the draft collector, for a second computer with its own Riot API key.

Same collector as scripts.collect_drafts, but it crawls Emerald players and keeps its own
database (data/collector/drafts_emerald.sqlite). Its games are sent back with `pack` and
added to the main database with `python -m scripts.collect_drafts merge <file>`.

Setup (once, needs Python 3.10+ and git):
    git clone https://github.com/jcj1337/lol_draft_project.git
    cd lol_draft_project
    git checkout data-collector
    python -m venv .venv
    .venv\\Scripts\\pip install requests pandas python-dotenv
    Create a file named .env in this folder containing one line: RIOT_API_KEY=RGAPI-...

Run from the project folder (on Mac/Linux use .venv/bin/python):
    .venv\\Scripts\\python -m scripts.collect_emerald           collect until Ctrl+C (resumable)
    .venv\\Scripts\\python -m scripts.collect_emerald status    what has been collected
    .venv\\Scripts\\python -m scripts.collect_emerald pack      file of new games to send back

The first start downloads the Emerald ladder, which takes about an hour on the biggest
servers; games start arriving as each server's ladder finishes. Development keys expire
every 24 hours: paste a new one into .env and the running collector resumes by itself.
"""
import sys

from scripts import collect_drafts

if __name__ == "__main__":
    collect_drafts.use_ladder("emerald")
    collect_drafts.main(sys.argv[1:] or ["collect"])
