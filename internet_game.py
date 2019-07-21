import random


class Domino:
    '''represents a single domino
    attribute:
      pips: a tuple representing the two pips'''

    def __init__(self,a,b):
        '''Domino(a,b) -> Domino
        creates the domino a-b'''
        self.pips = ((a,b))

    def __str__(self):
        '''str(Domino) -> str'''
        return str(self.pips[0])+'-'+str(self.pips[1])

    def reverse_str(self):
        '''Domino.reverse_str() -> str
        string with pips reversed'''
        return str(self.pips[1])+'-'+str(self.pips[0])

    def get_pips(self):
        '''Domino.get_pips() -> tuple
        returns the pips of the domino'''
        return self.pips

    def reverse(self):
        '''Domino.reverse()
        reverses the pips'''
        (a,b) = self.pips
        self.pips = (b,a)

    def is_match(self,pip):
        '''Domino.is_match(pip) -> int
        checks if the domino matches the given pip
        returns the index (0 or 1) of the half that matches
        returns -1 if not a match'''
        if self.pips[0]==pip:
            return 0
        elif self.pips[1]==pip:
            return 1
        else:
            return -1


class DominoSet():
    '''represents a full set of dominos'''

    def __init__(self):
        '''DominoSet() -> DominoSet'''
        self.dominoes = []
        for a in range(7):
            for b in range(a,7):
                self.dominoes.append(Domino(a,b))
        random.shuffle(self.dominoes)

    # __str__ doesn't seem necessary -- we never need a
    #  string rep for our DominoSet since it's created
    #  and then immediately consumed

    def deal(self):
        '''DominoSet.deal() -> list
        returns a list of 7 dominoes'''
        return [self.dominoes.pop() for i in range(7)]


class Chain:
    '''represents a domino chain
    attributes:
      chain: a list of dominoes'''

    def __init__(self):
        '''Chain() -> Chain
        start a new domino chain with the 6-6 domino'''
        self.chain = [Domino(6,6)]

    def __str__(self):
        '''str(Chain) -> str'''
        return ','.join([str(d) for d in self.chain])

    def left_end(self):
        '''Chain.left_end() -> int
        returns the number on the left end of the chain'''
        # return the first number on the first domino
        return self.chain[0].get_pips()[0]

    def right_end(self):
        '''Chain.right_end() -> int
        returns the number on the right end of the chain'''
        # return the second number on the last domino
        return self.chain[-1].get_pips()[1]

    def is_playable(self,domino):
        '''Chain.is_playable(Domino) -> boolean
        returns True if the domino can be played on either end
          of the chain, False if not'''
        # must check if it is a match at either end
        return domino.is_match(self.left_end())>=0 or \
               domino.is_match(self.right_end())>=0

    def add(self,domino,rightOnly=False):
        '''Chain.add(domino)
        adds a domino to the chain
        does nothing if the domino cannot be added
        if rightOnly is True, only plays on the right side'''
        # check both pips of the domino at both ends
        #  of the chain -- reverse if necessary before adding
        if not rightOnly and domino.is_match(self.left_end()) == 0:
            # left pip at left end
            domino.reverse()
            self.chain.insert(0,domino)
        elif not rightOnly and domino.is_match(self.left_end()) == 1:
            # right pip at left end
            self.chain.insert(0,domino)
        elif domino.is_match(self.right_end()) == 0:
            # left pip at right end
            self.chain.append(domino)
        elif domino.is_match(self.right_end()) == 1:
            # right pip at right end
            domino.reverse()
            self.chain.append(domino)


class Player:
    '''represents a dominoes player
    attributes:
      isHuman: True if human, False if computer
      hand: a Hand'''

    def __init__(self,isHuman,dominoes):
        '''Player(isHuman,dominoes) -> Player
        creates a new player with a 7-domino hand taken from dominoes
        isHuman is True for a human player, False for a computer player'''
        self.isHuman = isHuman
        self.hand = dominoes.deal()

    def __str__(self):
        '''str(Player) -> str'''
        if self.isHuman:
            return 'You have '+str(len(self.hand))+' dominoes'
        else:
            return 'A computer player has '+str(len(self.hand))+' dominoes'

    def is_human(self):
        '''Player.is_human() -> boolean
        returns True for a human player, False for a computer player'''
        return self.isHuman

    def goes_first(self):
        '''Player.goes_first() -> boolean
        makes first move and returns True if player has 6-6
        returns False otherwise'''
        for domino in self.hand:
            if domino.get_pips()[0] == 6 and domino.get_pips()[1] == 6:
                self.hand.remove(domino)
                return True
        return False

    def has_won(self):
        '''Player.has_won() -> boolean
        returns True if the player has won, False otherwise'''
        # player wins if his hand is empty
        return len(self.hand) == 0

    def take_turn(self,chain):
        '''Player.take_turn(chain) -> boolean
        takes the player's turn in the game
        chain is the current chain
        returns True if the player passes, False otherwise'''
        if self.isHuman:  # human player's turn
            # print the chain and the player's hand
            print("It's your turn.")
            print("The current chain:")
            print(chain)
            print("Your hand:")
            print('\n'.join([str(self.hand.index(d))+': '+str(d) for d in self.hand]))

            while True:  # get a choice of a domino to play, or pass
                choice = input("Which do you want to play? Enter p to pass. ")
                if choice.lower() == 'p':  # pass
                    break
                if choice.isdigit():  # wants to play
                    # validate the choice
                    if (int(choice) < 0 or int(choice) >= len(self.hand)):
                        print("Invalid domino number!")
                    elif not chain.is_playable(self.hand[int(choice)]):
                        print("That domino is not playable!")
                    else:
                        break

            if choice.isdigit():  # play
                choice = int(choice)
                domino = self.hand.pop(choice) # remove domino from hand
                rightOnly = False  # by default play on either side
                # if can play on either side, ask which side
                if domino.is_match(chain.left_end()) >= 0 and \
                   domino.is_match(chain.right_end()) >= 0:
                    print("That domino matches both sides of the chain.")
                    response = 'x'
                    while response.lower() not in 'lr':
                        response = input("Which side do you want to play it on? (Type l or r) ")
                    if response.lower() == 'r':
                        rightOnly = True
                chain.add(domino,rightOnly) # add domino to chain
                return False
            else:
                return True

        else:  # computer player
            # get list of dominos that the computer can play
            playlist = [d for d in self.hand if chain.is_playable(d)]

            if len(playlist) > 0:  # can play
                # pick a playable domino at random and play it
                domino = playlist[random.randrange(len(playlist))]
                self.hand.remove(domino)  # remove domino from hand
                chain.add(domino)  # add domino to chain
                print("Computer player plays "+str(domino)+", has "+\
                      str(len(self.hand))+" dominoes remaining.")
                return False

            else: # computer can't play
                print("Computer player passes"+", has "+\
                      str(len(self.hand))+" dominoes remaining.")
                return True

def play_solo() :
    '''play_solo() -> number
    plays dominoes with 1 human and 3 computer players
    returns player number of winner (human = 0)'''
    # create new set of dominoes and initialize the chain
    dominoes = DominoSet()
    chain = Chain()

    # create human player
    playerList = [Player(True,dominoes)]
    # create 3 computer players
    for i in range(3):
        playerList.append(Player(False,dominoes))

    # figure out which player goes first
    # first move is automatic, so set currentPlayerNum to the next player
    for i in range(4):
        if playerList[i].goes_first():
            if playerList[i].is_human():
                print("You went first by placing 6-6.")
            else:
                print("A computer player went first by placing 6-6.")
            currentPlayerNum = (i+1)%4  

    passCount = 0  # to keep track if there are 4 passes in a row
    while True:  # play the game
        player = playerList[currentPlayerNum] # the current player
        passed = player.take_turn(chain)  # take a turn
        if passed:
            passCount += 1  # increase the pass count
        else:
            passCount = 0  # reset the pass count
        if player.has_won() or passCount == 4:  # game is over
            # print winning message
            if player.is_human():
                print("You won!")
            else:
                print("Sorry, a computer player won.")
            return currentPlayerNum  #  end the game
        # go to the next player
        currentPlayerNum = (currentPlayerNum + 1) % 4

# play the game
play_solo()