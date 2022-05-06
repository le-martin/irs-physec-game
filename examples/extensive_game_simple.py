#%%
import pygambit

g = pygambit.Game.new_tree()
g.title = "Simple extensive game example"

bob = g.players.add("Bob")
eve = g.players.add("Eve")

move = g.root.append_move(bob, 2)
move.label = "First round: player Bob"
move.actions[0].label = 'A'
move.actions[1].label = 'B'

move = g.root.children[0].append_move(eve, 2)
move.label = "Second round: player Eve"
move.actions[0].label = 'D'
move.actions[1].label = 'E'

move = g.root.children[1].append_move(eve, 2)
move.label = "Second round: player Eve"
move.actions[0].label = 'F'
move.actions[1].label = 'G'

payoff_tuple = g.outcomes.add("5,2")
payoff_tuple[0] = 1
payoff_tuple[1] = 1
g.root.children[0].children[0].outcome = payoff_tuple

payoff_tuple = g.outcomes.add("3,1")
payoff_tuple[0] = 1
payoff_tuple[1] = 1
g.root.children[0].children[1].outcome = payoff_tuple

payoff_tuple = g.outcomes.add("6,3")
payoff_tuple[0] = 1
payoff_tuple[1] = 1
g.root.children[1].children[0].outcome = payoff_tuple

payoff_tuple = g.outcomes.add("4,4")
payoff_tuple[0] = 1
payoff_tuple[1] = 1
g.root.children[1].children[1].outcome = payoff_tuple

solver = pygambit.nash.ExternalEnumPureSolver()
nash_eq = solver.solve(g)
print(nash_eq[0]._profile)
print(nash_eq[1]._profile)

# Write game into a file
with open("extensive_game_simple_test.efg", "w") as f:
    f.write(g.write())
