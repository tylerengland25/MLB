from bs4 import BeautifulSoup
from urllib.request import urlopen
import pandas as pd


def scrape_season(season):
    # Connect to website
    url = f'https://www.baseball-reference.com/leagues/majors/{season}-schedule.shtml'
    html = urlopen(url)
    soup = BeautifulSoup(html, features="lxml")


    # Data structure for schedules
    data = {'date': [], 'visitor': [], 'home': []}
    for slate in soup.find('div', attrs={'class': 'section_content'}).find_all('div'):
        # Date of slate of games
        date = slate.find('h3').text
        print(f'\t{date}')
        for game in slate.find_all('p', attrs={'class': 'game'}):
            # Home and visitor team names
            teams = game.find_all('a')
            print(f'\t\t{teams[0].text} @ {teams[1].text}')

            # Append data
            data['date'].append(date)
            data['visitor'].append(teams[0].text)
            data['home'].append(teams[1].text)
    
    return data


def main():
    for season in range(2022 - 5, 2022):
        print(f'Season: {season}')
        df = pd.DataFrame(scrape_season(season))
        df['date'] = pd.to_datetime(df['date'])
        df.to_csv(f'backend/data/schedules/{season}.csv', index=False)



if __name__ == '__main__':
    main()
 