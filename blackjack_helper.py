import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DECK_NAME = np.array(["10", "A","2", "3", "4", "5", "6", "7", "8", "9"])
STARTING_DECK = np.array([16, 4, 4, 4, 4, 4, 4, 4, 4, 4])
DECK_VALUES = np.array([10, 11, 2, 3, 4, 5, 6, 7, 8, 9])

HAND_NAME = np.array(["P_S","P_A","D_S"]+list(DECK_NAME))

GAME_NAME = np.array(["Prob"]+list(HAND_NAME))

COLORS = {0:"tomato", 17:"orange", 18:"yellow", 19:"greenyellow", 20:"limegreen", 21:"deepskyblue"}

def get_deck(num_decks=4):
    return STARTING_DECK * num_decks

def print_deck(deck):
    assert type(deck) is np.ndarray, f"deck is not a numpy array, but {type(deck)}"
    assert deck.shape == STARTING_DECK.shape, f"deck has shape {deck.shape}, not {STARTING_DECK.shape}"

    df = pd.DataFrame([], columns=DECK_NAME)
    df.loc[0] = deck
    print(df)

def plot_deck(deck):
    assert type(deck) is np.ndarray, f"deck is not a numpy array, but {type(deck)}"
    assert deck.shape == STARTING_DECK.shape, f"deck has shape {deck.shape}, not {STARTING_DECK.shape}"

    plt.figure(figsize=(5, 2))
    plt.title("Card Distribution")
    plt.bar(DECK_NAME, deck)
    plt.xticks(np.arange(len(DECK_NAME)), DECK_NAME)
    plt.show()

def deck_remove_card(deck, card):
    deck = deck.copy()
    deck[card%10] -= 1
    return deck

def get_hand(deck=get_deck(), dealer_card=None, player_cards=[]):
    deck = deck.copy()
    if dealer_card is not None:
        deck = deck_remove_card(deck, dealer_card)
    else:
        dealer_card = 0
    
    player_score = 0
    player_aces = 0
    for card in player_cards:
        deck = deck_remove_card(deck, card)
        if card == 11:
            player_aces += 1
        player_score += card
    
    # TODO: efficiency
    while player_score > 21 and player_aces > 0:
        player_score -= 10
        player_aces -= 1

    return np.array([player_score,player_aces,dealer_card] + list(deck))

def print_hand(hand):
    assert type(hand) is np.ndarray, f"hand is not a numpy array, but {type(hand)}"
    assert hand.shape == HAND_NAME.shape, f"hand has shape {hand.shape}, not {HAND_NAME.shape}"

    df = pd.DataFrame([], columns=HAND_NAME)
    df.loc[0] = hand
    print(df)


def get_game(deck=get_deck(), dealer_card=None, player_cards=[]):
    hand = get_hand(deck, dealer_card, player_cards)
    game = np.array([[1.0]+list(hand)])
    return game

def print_game(game):
    assert type(game) is np.ndarray, f"game is not a numpy array, but {type(game)}"

    rounds = game.shape[1] - 14
    round_columns = [f"R{i}" for i in range(1,rounds+1)]

    df = pd.DataFrame(game, columns=list(GAME_NAME)+round_columns).astype(
        dict(zip(
            list(GAME_NAME)+round_columns,
            [float]+[int]*(len(HAND_NAME)+len(round_columns))
        ))
    )
    print(df)

def game_dealer_first_card(game, card=None):
    assert game.shape == (1, len(GAME_NAME)), f"game has shape {game.shape}, not (1, {len(GAME_NAME)})"

    game = game.copy()
    if card != None:
        game[0,3] = card
        game[0,4:14] = deck_remove_card(game[0,4:14], card)
    else:
        game = np.tile(game, (10,1))
        game[:,3] = DECK_VALUES

        # update probabilties
        game[:,0] *= (np.diagonal(game[:,4:14]) / (game[:,4:14]).sum(axis=1))

        # update deck
        game[:,4:14] = game[:,4:14] - np.diag(np.ones(10))
    return game

def game_player_hit(game, card=None):
    assert len(game.shape) == 2, f"game is not a 2D array, but {len(game.shape)}D"
    assert game.shape[1] >= len(GAME_NAME), f"game has less than {game.shape[1]} columns"
    assert np.all(game[:,3] > 0), f"dealer has not yet drawn a card"

    game_new = game.copy()
    orig_shape = game.shape

    # add indicator for standing
    game = np.hstack((game, np.zeros((game.shape[0], 1))))

    # add indicator for hitting
    game_new = np.hstack((game_new, np.ones((game_new.shape[0], 1))))  

    if card != None:
        game_new[:,1] += (card%10)
        game_new[:,2] += (card%10 == 11).astype(int)
        game_new[:,4+(card%10)] -= 1

        game_new[:,0] *= (game_new[:,4+(card%10)] / (game_new[:,4:14]).sum(axis=1))
    else:
        game_new = np.repeat(game_new, 10, axis=0)
        # print(game_new, game_new.shape)
        game_new[:,2] += np.tile(DECK_VALUES == 11, orig_shape[0]).astype(int)
        game_new[:,1] += np.tile(DECK_VALUES, orig_shape[0])
        game_new[:,4:14] = game_new[:,4:14] - np.tile(np.diag(np.ones(10)), (orig_shape[0],1))

        # TODO: scaling with tile / repeat
        game_new[:,0] *= (np.diagonal(game_new[:,4:14]) / (game_new[:,4:14]).sum(axis=1))
    
    while np.any(np.logical_and(game_new[:,1] > 21,game_new[:,2] > 0)):
        # print_game(game_new[np.logical_and(game_new[:,1] > 21,game_new[:,2] > 0), :])
        game_new[np.logical_and(game_new[:,1] > 21,game_new[:,2] > 0), 1:3] -= (10,1)

    # TODO: bust handling?

    # combine hitting and standing
    game = np.vstack((game, game_new))

    return game

def get_dealer_hands(cards=[]):
    finished_hands = pd.DataFrame([])

    if len(cards) == 0:
        possible_hands = np.array([[0]])
    else:
        possible_hands = np.array([[int(np.sum(np.array(cards) == 11))]+list(cards)]).T.reshape(1,-1)

    scores = possible_hands[:,1:].sum(axis=1)
    scores[scores > 21] -= np.min((
        possible_hands[(scores > 21),0], #10
        ((scores[(scores > 21)]-12)//10) #126 // 10 = 12
    ), axis=0)*10
    # print(scores)

    new_finished_hands = possible_hands[scores >= 17,:]
    possible_hands = possible_hands[scores < 17,:]

    finished_hands = pd.concat([finished_hands, pd.DataFrame(new_finished_hands)], axis=0).fillna(0).astype(int)


    while len(possible_hands) > 0:
        possible_hands = np.hstack((
            np.tile(possible_hands, len(DECK_VALUES)).reshape(-1, possible_hands.shape[1]),
            np.tile(DECK_VALUES, possible_hands.shape[0]).reshape(-1, 1)))
        possible_hands[:,0] += (possible_hands[:,-1] == 11).astype(int)
        # print(f"Possible Hands Shape: {possible_hands.shape}")
        # print(possible_hands[:5,:])

        scores = possible_hands[:,1:].sum(axis=1)
        scores[scores > 21] -= np.min((
            possible_hands[(scores > 21),0], #10
            ((scores[(scores > 21)]-12)//10) #126 // 10 = 12
        ), axis=0)*10
        # print(scores)

        new_finished_hands = possible_hands[scores >= 17,:]
        possible_hands = possible_hands[scores < 17,:]

        finished_hands = pd.concat([finished_hands, pd.DataFrame(new_finished_hands)], axis=0).fillna(0).astype(int)

    finished_scores = finished_hands.iloc[:,1:].sum(axis=1)
    finished_scores[finished_scores > 21] -= np.min((
        finished_hands.loc[(finished_scores > 21),0], #10
        ((finished_scores[(finished_scores > 21)]-12)//10) #126 // 10 = 12
    ), axis=0)*10
    finished_scores[finished_scores > 21] = 0

    finished_hands["Score"] = finished_scores
    finished_hands.rename(columns={0:"Aces"}, inplace=True)

    return finished_hands.reset_index(drop=True)

def get_dealer_possibilities(cards=[]):
    finished_hands = get_dealer_hands(cards)

    res = pd.DataFrame(index=[0, 17, 18, 19, 20, 21], columns=["Count"]).fillna(0)
    res["Count"] = finished_hands["Score"].value_counts().sort_index()
    res["Count"] = res["Count"].fillna(0).astype(int)

    # res = pd.DataFrame({"Count":finished_hands["Score"].value_counts().sort_index()})
    return res

def plot_dealer_possibilities(cards=[]):
    res = get_dealer_possibilities(cards)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [1, 2]})

    res.plot(kind="pie", ax=axes[0], autopct='%1.2f%%', startangle=90, ylabel="", subplots=True, legend=False, colors=list(COLORS.values()))
    axes[1].bar(res.index.astype(str), res["Count"], width=0.9, color=[COLORS[x] for x in res.index])

    fig.suptitle(f"Dealer Scores Analysis with cards: {cards}", fontsize=16)

def update_deck_with_cards(deck, cards):
    deck = deck.copy()
    for card in cards:
        deck[card%10] -= 1
    return deck

def get_dealer_hand_probabilities(cards=[], deck=get_deck()):
    assert deck.shape == (10,), f"deck has shape {deck.shape}, not (10,)"

    possible_hands_prob = get_dealer_hands(cards).copy()
    possible_hands_prob["Probability"] = None

    for i, row in possible_hands_prob.iterrows():
        score = row["Score"]
        row = row.values[1:-2]
        row = row[row > 0]
        # print(score, row)

        deck_copy = deck.copy()
        probability = 1.0

        for element in row:
            # print(f"el {element%10} with {deck_copy[element%10]} left. Prob: {deck_copy[element%10] / deck_copy.sum()}")
            probability *= deck_copy[element%10] / deck_copy.sum()
            if deck_copy[element%10] > 0:
                deck_copy[element%10] -= 1
            else:
                break
        possible_hands_prob.loc[i, "Probability"] = probability
                    
    return possible_hands_prob

def get_dealer_score_probabilities(cards=[], deck=get_deck()):
    assert deck.shape == (10,), f"deck has shape {deck.shape}, not (10,)"

    possible_hands_prob = get_dealer_hand_probabilities(cards, deck)
    scores_prob = possible_hands_prob.groupby("Score").agg({"Probability": "sum"})
    return scores_prob

def plot_dealer_score_probabilities(cards=[], deck=get_deck()):
    assert deck.shape == (10,), f"deck has shape {deck.shape}, not (10,)"

    scores_prob = get_dealer_score_probabilities(cards, deck)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [1, 2]})

    scores_prob.plot(kind="pie", ax=axes[0], autopct='%1.2f%%', startangle=90, ylabel="", subplots=True, legend=False, colors=list(COLORS.values()))
    axes[1].bar(scores_prob.index.astype(str), scores_prob["Probability"], width=0.9, color=[COLORS[x] for x in scores_prob.index])

    fig.suptitle(f"Dealer Scores Probabilities with cards: {cards}", fontsize=16)

def get_win_lose_prob(game):
    assert game.shape == (1, len(GAME_NAME)), f"game has shape {game.shape}, not (1, {len(GAME_NAME)})"

    dealer_card = list(game[0,3:4])
    deck = game[0,4:14]

    player_score = game[0,1]

    scores_prob = get_dealer_score_probabilities(dealer_card, deck)

    win = scores_prob.loc[scores_prob.index < player_score,"Probability"].sum()
    lose = scores_prob.loc[scores_prob.index > player_score,"Probability"].sum()
    draw = scores_prob.loc[scores_prob.index == player_score,"Probability"].sum()

    # print(f"Win: {win:.2f}, Lose: {lose:.2f}, Draw: {draw:.2f}")

    return pd.DataFrame({"Win":win, "Lose":lose, "Draw":draw}, index=[0])


if __name__ == "__main__":
    pass