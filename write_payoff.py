#%%
import numpy as np

file_read = "extensive_game_wo_rewards.efg"
payoff_arr = np.load("r_s.npy")

f = open(file_read, "r")
lines_read = f.readlines()
f.close()

# Create list with line numbers to change in game file and new lines of file
lines_with_changes = []
lines_write = lines_read
cnt = 4
for i in range(625):
    cnt += 1
    for j in range(625):
        lines_with_changes.append(cnt)
        idx_tree = i*625+j
        line_new = f't "" {idx_tree} "{payoff_arr[idx_tree, 0]}" {{ {payoff_arr[idx_tree, 0]},' \
                   f'{-payoff_arr[idx_tree, 0]} }}\n'
        lines_write[cnt] = line_new
        cnt += 1

f = open(f'{file_read.rsplit(".")[0]}_changed.efg', "w")
f.writelines(lines_write)
f.close()

print("Done")
