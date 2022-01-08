
def dynamicProgramming(dp):
    x = len(dp)
    y = len(dp[0])
    dp[0][0] = 984
    dp[x-1][y-1] = 621

dp = [[-1] * 2 for i in range(5)]
dynamicProgramming(dp)


