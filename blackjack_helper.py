import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DECK_NAME = np.array(["2", "3", "4", "5", "6", "7", "8", "9", "10", "A"])
STARTING_DECK = np.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 16])
DECK_VALUES = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

HAND_NAME = np.array(["P_S","P_A","D_S","D_A","2","3","4","5","6","7","8","9","10","A"])

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

def get_hand(deck=get_deck()):
    return np.array([0,0,0,0] + list(deck))

def print_hand(hand):
    assert type(hand) is np.ndarray, f"hand is not a numpy array, but {type(hand)}"
    assert hand.shape == HAND_NAME.shape, f"hand has shape {hand.shape}, not {HAND_NAME.shape}"

    df = pd.DataFrame([], columns=HAND_NAME)
    df.loc[0] = hand
    print(df)

def get_dealer_possibilities(cards=[]):
    finished_hands = pd.DataFrame([])

    if len(cards) == 0:
        possible_hands = np.array([[0]])
    else:
        possible_hands = np.array([(np.array(cards) == 11).astype(int), cards]).T

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

    res = pd.DataFrame({"Count":finished_hands["Score"].value_counts().sort_index()})
    return res

def plot_dealer_possibilities(cards=[]):
    res = get_dealer_possibilities(cards)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [1, 2]})

    res.plot(kind="pie", ax=axes[0], autopct='%1.2f%%', startangle=90, ylabel="", subplots=True, legend=False, colors=list(COLORS.values()))
    axes[1].bar(res.index.astype(str), res["Count"], width=0.9, color=[COLORS[x] for x in res.index])

    fig.suptitle("Dealer Scores Analysis", fontsize=16)

if __name__ == "__main__":
    plot_dealer_possibilities()