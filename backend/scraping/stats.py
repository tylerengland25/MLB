from bs4 import BeautifulSoup
import bs4
from urllib.request import urlopen
import pandas as pd


def scrape_scorebox(date, teams, table):
    innings = [th.text for th in table.find_all('tr')[0].find_all('th')][2:]
    visitor_scores = [td.text for td in table.find_all('tr')[1].find_all('td')][1:]
    home_scores = [td.text for td in table.find_all('tr')[2].find_all('td')][1:]
    
    df = pd.DataFrame(
        columns = ['date', 'visitor', 'home', 'team'] + innings
    ).append(
        pd.Series(
            [date, teams[0], teams[1]] + visitor_scores, 
            index=['date', 'visitor', 'home', 'team'] + innings
        ),
        ignore_index=True
    ).append(
        pd.Series(
            [date, teams[0], teams[1]] + home_scores, 
            index=['date', 'visitor', 'home', 'team'] + innings
        ),
        ignore_index=True
    )

    return df


def scrape_batting_details(date, teams, team, table):
    cols = [th.text.lower() for th in table.find('tr').find_all('th')][1:]
    df = pd.DataFrame(
        columns=['date', 'visitor', 'home', 'team', 'player', 'position'] + cols
    )

    for row in table.find_all('tr')[1:]:
        player = " ".join(x.strip() for x in row.find('th').text.split(" ")[:-1])
        position = "".join(row.find('th').text.split(" ")[-1])
        data = [td.text for td in row.find_all('td')]

        if player != 'Team' and player != "":
            df = df.append(
                pd.Series(
                    [date, teams[0], teams[1], team, player, position] + data,
                    index=df.columns
                ), 
                ignore_index=True
            )

    return df

    
def scrape_batting_totals(date, teams, team, table):
    cols = [th.text.lower() for th in table.find('tr').find_all('th')][1:]
    df = pd.DataFrame(
        columns=['date', 'visitor', 'home', 'team'] + cols
    )

    for row in table.find_all('tr')[1:]:
        player = " ".join(row.find('th').text.split(" ")[:-1])
        data = [td.text for td in row.find_all('td')]

        if player == 'Team':
            df = df.append(
                pd.Series(
                    [date, teams[0], teams[1], team] + data,
                    index=df.columns
                ), 
                ignore_index=True
            )

    return df


def scrape_pitching_details(date, teams, team, table):
    cols = [th.text.lower() for th in table.find('tr').find_all('th')][1:]
    df = pd.DataFrame(
        columns=['date', 'visitor', 'home', 'team', 'player'] + cols
    )

    for row in table.find_all('tr')[1:]:
        player = " ".join(x.strip() for x in row.find('th').text.split(" ")[:2])
        data = [td.text for td in row.find_all('td')]

        if player != 'Team Totals' and player != "":
            df = df.append(
                pd.Series(
                    [date, teams[0], teams[1], team, player] + data,
                    index=df.columns
                ), 
                ignore_index=True
            )

    return df


def scrape_pitching_totals(date, teams, team, table):
    cols = [th.text.lower() for th in table.find('tr').find_all('th')][1:]
    df = pd.DataFrame(
        columns=['date', 'visitor', 'home', 'team'] + cols
    )

    for row in table.find_all('tr')[1:]:
        player = " ".join(row.find('th').text.split(" ")[:2])
        data = [td.text for td in row.find_all('td')]

        if player == 'Team Totals':
            df = df.append(
                pd.Series(
                    [date, teams[0], teams[1], team] + data,
                    index=df.columns
                ), 
                ignore_index=True
            )

    return df


def scrape_game(data, date, teams, link):
    # Connect to boxscore websit
    url = f'https://www.baseball-reference.com{link}'
    html = urlopen(url)
    soup = BeautifulSoup(html, features="lxml")

    # Boxscores of innings
    scores = soup.find('div', attrs={'class': 'linescore_wrap'}).find('table')

    # Scape tables that are commented
    visitor_batting = None
    visitor_pitching = None
    home_batting = None
    home_pitching = None

    for comment in soup.find_all(text=lambda text: isinstance(text, bs4.Comment)):
        if comment.find("<table ") > 0:
            comment_soup = BeautifulSoup(comment, 'html.parser')
            if visitor_batting is None:
                visitor_batting = comment_soup.find('table' , {'id' : f'{"".join([x.strip(".") for x in teams[0].split(" ")])}batting'})
            if visitor_pitching is None:
                visitor_pitching = comment_soup.find('table' , {'id' : f'{"".join([x.strip(".") for x in teams[0].split(" ")])}pitching'})
            if home_batting is None:
                home_batting = comment_soup.find('table' , {'id' : f'{"".join([x.strip(".") for x in teams[1].split(" ")])}batting'})
            if home_pitching is None:
                home_pitching = comment_soup.find('table' , {'id' : f'{"".join([x.strip(".") for x in teams[1].split(" ")])}pitching'})

    # Scapre data from tables
    data['boxscore'] = data['boxscore'].append(scrape_scorebox(date, teams, scores), ignore_index=True)

    data['batting_details'] = data['batting_details'].append(scrape_batting_details(date, teams, teams[0], visitor_batting), ignore_index=True)
    data['batting_details'] = data['batting_details'].append(scrape_batting_details(date, teams, teams[1], home_batting), ignore_index=True)
    data['pitching_details'] = data['pitching_details'].append(scrape_pitching_details(date, teams, teams[0], visitor_pitching), ignore_index=True)
    data['pitching_details'] = data['pitching_details'].append(scrape_pitching_details(date, teams, teams[1], home_pitching), ignore_index=True)

    data['batting_totals'] = data['batting_totals'].append(scrape_batting_totals(date, teams, teams[0], visitor_batting), ignore_index=True)
    data['batting_totals'] = data['batting_totals'].append(scrape_batting_totals(date, teams, teams[1], home_batting), ignore_index=True)
    data['pitching_totals'] = data['pitching_totals'].append(scrape_pitching_totals(date, teams, teams[0], visitor_pitching), ignore_index=True)
    data['pitching_totals'] = data['pitching_totals'].append(scrape_pitching_totals(date, teams, teams[1], home_pitching), ignore_index=True)


def scrape_season(season, data):
    # Connect to website
    url = f'https://www.baseball-reference.com/leagues/majors/{season}-schedule.shtml'
    html = urlopen(url)
    soup = BeautifulSoup(html, features="lxml")

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

            # Fix D'backs name
            teams[0] = 'Arizona Diamondbacks' if teams[0] == 'Arizona D\'Backs' else teams[0]
            teams[1] = 'Arizona Diamondbacks' if teams[1] == 'Arizona D\'Backs' else teams[1]

            # href Link
            link = game.find('em').find('a')['href']
            # Scrape game data
            scrape_game(data, date, teams, link)
    
    return data


def main():
    # Data structure for batting details and totals, pitching details and totals
    data = {
        'boxscore': pd.DataFrame(),
        'batting_details': pd.DataFrame(), 
        'batting_totals': pd.DataFrame(), 
        'pitching_details': pd.DataFrame(),
        'pitching_totals': pd.DataFrame()
        }


    for season in range(2022 - 5, 2022):
        print(f'Season: {season}')
        data = scrape_season(season, data)
        for df in data:
            data[df]['date'] = pd.to_datetime(data[df]['date'])
            if len(df.split("_")) > 1:

                data[df].to_csv(f'backend/data/{df.split("_")[0]}/{df.split("_")[1]}.csv', index=False)
            else:
                data[df].to_csv(f'backend/data/scores/{df}.csv', index=False)


if __name__ == '__main__':
    main()
 