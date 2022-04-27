import datetime
from bs4 import BeautifulSoup
from urllib.request import urlopen
import pandas as pd


def scrape_season(current_season):
    # Connect to website
    url = f'https://www.baseball-reference.com/leagues/majors/{current_season}-schedule.shtml'
    html = urlopen(url)
    soup = BeautifulSoup(html, features="lxml")


    # Data structure for schedules
    data = {'date': [], 'visitor': [], 'home': [], 'season': []}
    
    # Regular and Post season
    slates = []
    for season in soup.find_all('div', attrs={'class': 'section_content'}):
        for slate in season.find_all('div'):
            slates.append(slate)

    for slate in slates:
        # Date of slate of games
        date = slate.find('h3').text
        print(f'\t{date}')
        for game in slate.find_all('p', attrs={'class': 'game'}):
            # Home and visitor team names
            teams = [team.text for team in game.find_all('a')]
            print(f'\t\t{teams[0]} @ {teams[1]}')

            # Today's games
            if date == "Today's Games":
                date = datetime.date.today()

            # Append data
            data['date'].append(date)
            data['visitor'].append(teams[0])
            data['home'].append(teams[1])
            data['season'].append(current_season)
    
    return data


def main():
    current_season = 2022
    for season in range(current_season - 5, current_season + 1):
        print(f'Season: {season}')
        df = pd.DataFrame(scrape_season(season))
        df['date'] = pd.to_datetime(df['date'])
        df.to_csv(f'backend/data/schedules/{season}.csv', index=False)


if __name__ == '__main__':
    main()
 