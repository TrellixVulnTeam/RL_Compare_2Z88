####### CHARTS FOR VALUE ITERATION ########
# Iterations vs Time plot for value iteration
times = []
iters = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 1000000]
for iter in iters:
    tic = time.perf_counter()
    # Generate a 10x10 map with 0.7 probility tile is slippery.
    pols = val_iter(rand_map_16, GAMMA, FROZEN_LAKE, iters=iter)
    opt_pol = find_best_policy(rand_map_16,
                               GAMMA,
                               pols,
                               FROZEN_LAKE)
    # ic(opt_pol)
    toc = time.perf_counter()
    print(f'Took {toc - tic} seconds')
    times.append(toc - tic)

df_for_vl = pd.DataFrame(data={'Iterations': iters,
                               'Time(seconds)': times})
ic(df_for_vl.dtypes)
# Do the time plot
sns.lineplot(x='Iterations',
             y='Time(seconds)',
             data=df_for_vl,
             palette='pastel')
sns.set_style('dark')

plt.title(f'Time vs Iterations for Frozen Lake Value Iteration', fontsize=13)
plt.legend(loc='upper right')
plt.show()
plt.savefig('outputs/fl_vi_iter_time.png')

# Gammas vs Time plot for value iteration
times = []
# iters = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 1000000]
gammas = np.arange(0.05, 1.0, 0.05)
for i in gammas:
    tic = time.perf_counter()
    pols = val_iter(rand_map_16, i, FROZEN_LAKE, iters=NO_OF_ITERS)
    opt_pol = find_best_policy(rand_map_16,
                               i,
                               pols,
                               FROZEN_LAKE)
    # ic(opt_pol)
    toc = time.perf_counter()
    print(f'Took {toc - tic} seconds')
    times.append(toc - tic)

df_for_vl = pd.DataFrame(data={'Discounts': gammas,
                               'Time(seconds)': times})
ic(df_for_vl.dtypes)
# Do the time plot
sns.lineplot(x='Discounts',
             y='Time(seconds)',
             data=df_for_vl,
             palette='pastel')
sns.set_style('dark')

plt.title(f'Time vs Discounts for Frozen Lake Value Iteration', fontsize=13)
plt.legend(loc='upper right')
plt.show()
plt.savefig('outputs/fl_vi_gamma_time.png')

####### CHARTS FOR POLICY ITERATION ########
# Iterations vs Time plot for policy iteration
times = []
iters = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 1000000]
for iter in iters:
    tic = time.perf_counter()
    pol = pol_iter(rand_map_16, FROZEN_LAKE, GAMMA, iters=iter)
    # ic(opt_pol)
    toc = time.perf_counter()
    print(f'Took {toc - tic} seconds')
    times.append(toc - tic)

df_for_vl = pd.DataFrame(data={'Iterations': iters,
                               'Time(seconds)': times})
ic(df_for_vl.dtypes)
# Do the time plot
sns.lineplot(x='Iterations',
             y='Time(seconds)',
             data=df_for_vl,
             palette='pastel')
sns.set_style('dark')

plt.title(f'Time vs Iterations for Frozen Lake Policy Iteration', fontsize=13)
# plt.legend(loc='upper right')
plt.show()
plt.savefig('outputs/fl_pi_iters_time.png')

# Gammas vs Time plot for policy iteration
times = []
# iters = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 1000000]
gammas = np.arange(0.05, 1.0, 0.05)
for i in gammas:
    tic = time.perf_counter()
    pol = pol_iter(rand_map_16, FROZEN_LAKE, i, NO_OF_ITERS)
    # ic(opt_pol)
    toc = time.perf_counter()
    print(f'Took {toc - tic} seconds')
    times.append(toc - tic)

df_for_vl = pd.DataFrame(data={'Discounts': gammas,
                               'Time(seconds)': times})
ic(df_for_vl.dtypes)
# Do the time plot
sns.lineplot(x='Discounts',
             y='Time(seconds)',
             data=df_for_vl,
             palette='pastel')
sns.set_style('dark')

plt.title(f'Time vs Discounts for Frozen Lake Policy Iteration', fontsize=13)
# plt.legend(loc='upper right')
plt.show()
plt.savefig('outputs/fl_pi_gamma_time.png')
