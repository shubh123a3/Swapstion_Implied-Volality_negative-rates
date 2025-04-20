import numpy as np
import enum
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.integrate as integrate
import scipy.optimize as optimize
from scipy import interpolate


# This class defines puts and calls
class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0


def GeneratePathsHWEuler(NoOfPaths, NoOfSteps, T, P0T, lambd, eta):
    # time-step needed for differentiation
    dt = 0.0001
    f0T = lambda t: - (np.log(P0T(t + dt)) - np.log(P0T(t - dt))) / (2 * dt)

    # Initial interest rate is a forward rate at time t->0
    r0 = f0T(0.00001)
    theta = lambda t: 1.0 / lambd * (f0T(t + dt) - f0T(t - dt)) / (2.0 * dt) + f0T(t) + eta * eta / (
            2.0 * lambd * lambd) * (1.0 - np.exp(-2.0 * lambd * t))

    # theta = lambda t: 0.1 +t -t
    # print("changed theta")

    Z = np.random.normal(0.0, 1.0, [NoOfPaths, NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps + 1])
    R = np.zeros([NoOfPaths, NoOfSteps + 1])
    R[:, 0] = r0
    time = np.zeros([NoOfSteps + 1])

    dt = T / float(NoOfSteps)
    for i in range(0, NoOfSteps):
        # making sure that samples from normal have mean 0 and variance 1
        if NoOfPaths > 1:
            Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])
        W[:, i + 1] = W[:, i] + np.power(dt, 0.5) * Z[:, i]
        R[:, i + 1] = R[:, i] + lambd * (theta(time[i]) - R[:, i]) * dt + eta * (W[:, i + 1] - W[:, i])
        time[i + 1] = time[i] + dt

    # Outputs
    paths = {"time": time, "R": R}
    return paths


def HW_theta(lambd, eta, P0T):
    dt = 0.0001
    f0T = lambda t: - (np.log(P0T(t + dt)) - np.log(P0T(t - dt))) / (2 * dt)
    theta = lambda t: 1.0 / lambd * (f0T(t + dt) - f0T(t - dt)) / (2.0 * dt) + f0T(t) + eta * eta / (
            2.0 * lambd * lambd) * (1.0 - np.exp(-2.0 * lambd * t))
    # print("CHANGED THETA")
    return theta  # lambda t: 0.1+t-t


def HW_A(lambd, eta, P0T, T1, T2):
    tau = T2 - T1
    zGrid = np.linspace(0.0, tau, 250)
    B_r = lambda tau: 1.0 / lambd * (np.exp(-lambd * tau) - 1.0)
    theta = HW_theta(lambd, eta, P0T)
    temp1 = lambd * np.trapz(theta(T2 - zGrid) * B_r(zGrid), zGrid)

    temp2 = eta * eta / (4.0 * np.power(lambd, 3.0)) * (
            np.exp(-2.0 * lambd * tau) * (4 * np.exp(lambd * tau) - 1.0) - 3.0) + eta * eta * tau / (
                    2.0 * lambd * lambd)

    return temp1 + temp2


def HW_B(lambd, eta, T1, T2):
    return 1.0 / lambd * (np.exp(-lambd * (T2 - T1)) - 1.0)


def HW_ZCB(lambd, eta, P0T, T1, T2, rT1):
    B_r = HW_B(lambd, eta, T1, T2)
    A_r = HW_A(lambd, eta, P0T, T1, T2)
    return np.exp(A_r + B_r * rT1)


def HWMean_r(P0T, lambd, eta, T):
    # time-step needed for differentiation
    dt = 0.0001
    f0T = lambda t: - (np.log(P0T(t + dt)) - np.log(P0T(t - dt))) / (2.0 * dt)
    # Initial interest rate is a forward rate at time t->0
    r0 = f0T(0.00001)
    theta = HW_theta(lambd, eta, P0T)
    zGrid = np.linspace(0.0, T, 2500)
    temp = lambda z: theta(z) * np.exp(-lambd * (T - z))
    r_mean = r0 * np.exp(-lambd * T) + lambd * np.trapz(temp(zGrid), zGrid)
    return r_mean


def HW_r_0(P0T, lambd, eta):
    # time-step needed for differentiation
    dt = 0.0001
    f0T = lambda t: - (np.log(P0T(t + dt)) - np.log(P0T(t - dt))) / (2 * dt)
    # Initial interest rate is a forward rate at time t->0
    r0 = f0T(0.00001)
    return r0


def HW_Mu_FrwdMeasure(P0T, lambd, eta, T):
    # time-step needed for differentiation
    dt = 0.0001
    f0T = lambda t: - (np.log(P0T(t + dt)) - np.log(P0T(t - dt))) / (2 * dt)
    # Initial interest rate is a forward rate at time t->0
    r0 = f0T(0.00001)
    theta = HW_theta(lambd, eta, P0T)
    zGrid = np.linspace(0.0, T, 500)

    theta_hat = lambda t, T: theta(t) + eta * eta / lambd * 1.0 / lambd * (np.exp(-lambd * (T - t)) - 1.0)

    temp = lambda z: theta_hat(z, T) * np.exp(-lambd * (T - z))

    r_mean = r0 * np.exp(-lambd * T) + lambd * np.trapz(temp(zGrid), zGrid)

    return r_mean


def HWVar_r(lambd, eta, T):
    return eta * eta / (2.0 * lambd) * (1.0 - np.exp(-2.0 * lambd * T))


def HWDensity(P0T, lambd, eta, T):
    r_mean = HWMean_r(P0T, lambd, eta, T)
    r_var = HWVar_r(lambd, eta, T)
    return lambda x: st.norm.pdf(x, r_mean, np.sqrt(r_var))


def HW_CapletFloorletPrice(CP, N, K, lambd, eta, P0T, T1, T2):
    if CP == OptionType.CALL:
        N_new = N * (1.0 + (T2 - T1) * K)
        K_new = 1.0 + (T2 - T1) * K
        caplet = N_new * HW_ZCB_CallPutPrice(OptionType.PUT, 1.0 / K_new, lambd, eta, P0T, T1, T2)
        value = caplet
    elif CP == OptionType.PUT:
        N_new = N * (1.0 + (T2 - T1) * K)
        K_new = 1.0 + (T2 - T1) * K
        floorlet = N_new * HW_ZCB_CallPutPrice(OptionType.CALL, 1.0 / K_new, lambd, eta, P0T, T1, T2)
        value = floorlet
    return value


def HW_ZCB_CallPutPrice(CP, K, lambd, eta, P0T, T1, T2):
    B_r = HW_B(lambd, eta, T1, T2)
    A_r = HW_A(lambd, eta, P0T, T1, T2)

    mu_r = HW_Mu_FrwdMeasure(P0T, lambd, eta, T1)
    v_r = np.sqrt(HWVar_r(lambd, eta, T1))

    K_hat = K * np.exp(-A_r)
    a = (np.log(K_hat) - B_r * mu_r) / (B_r * v_r)

    d1 = a - B_r * v_r
    d2 = d1 + B_r * v_r

    term1 = np.exp(0.5 * B_r * B_r * v_r * v_r + B_r * mu_r) * st.norm.cdf(d1) - K_hat * st.norm.cdf(d2)
    value = P0T(T1) * np.exp(A_r) * term1

    if CP == OptionType.CALL:
        return value
    elif CP == OptionType.PUT:
        return value - P0T(T2) + K * P0T(T1)


def BS_Call_Put_Option_Price(CP, S_0, K, sigma, tau, r):
    if isinstance(K, list):
        K = np.array(K).reshape([len(K), 1])
    d1 = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma, 2.0)) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    if CP == OptionType.CALL:
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * tau)
    elif CP == OptionType.PUT:
        value = st.norm.cdf(-d2) * K * np.exp(-r * tau) - st.norm.cdf(-d1) * S_0
    return value


# Solve for Jamshidian's rate r* for the swap
def psi_sum(r, lambd, eta, P0T, Tm, payment_times, accruals):
    # sum of psi_i(r) = sum c_k * exp(A + B*r)
    s = 0.0
    for ck, Tk in zip(accruals, payment_times):
        s += ck * np.exp(HW_A(lambd, eta, P0T, Tm, Tk) + HW_B(lambd, eta, Tm, Tk) * r)
    return s


def jamshidian_root(lambd, eta, P0T, Tm, payment_times, accruals, K_swap, r0):
    # solve psi_sum(r*) = 1/K_swap
    func = lambda r: psi_sum(r, lambd, eta, P0T, Tm, payment_times, accruals) - 1.0 / K_swap

    # Add more robustness to root finding
    try:
        r_star = optimize.newton(func, r0, tol=1e-6, maxiter=100)
        # Verify convergence
        if abs(func(r_star)) > 1e-4:
            print(f"Warning: Jamshidian root finding did not converge properly. Error: {func(r_star)}")
            # Fallback to a more robust method
            r_star = optimize.brentq(func, -0.5, 0.5)
    except:
        print("Newton method failed, trying brentq instead")
        try:
            r_star = optimize.brentq(func, -0.5, 0.5)
        except:
            print("Both root finding methods failed. Using initial guess.")
            r_star = r0

    # Verify that we have indeed found the root
    residual = func(r_star)
    print(f"r* = {r_star}, Residual = {residual}")

    return r_star


# Price swaption via Jamshidian decomposition
def price_swaption(CP, lambd, eta, P0T, Tm, payment_times, accruals, K_swap, N):
    # get initial short rate via consistent HW function, not shadowed by float
    r0 = HW_r_0(P0T, lambd, eta)

    # 1) compute Jamshidian rate r*
    r_star = jamshidian_root(lambd, eta, P0T, Tm, payment_times, accruals, K_swap, r0)

    # 2) decompose into ZCB options
    total = 0.0
    for ck, Tk in zip(accruals, payment_times):
        A_r = HW_A(lambd, eta, P0T, Tm, Tk)
        B_r = HW_B(lambd, eta, Tm, Tk)
        Kk = np.exp(A_r + B_r * r_star)
        # Payer swaption (CALL) => PUT on bond; Receiver (PUT) => CALL on bond
        opt_type = OptionType.PUT if CP == OptionType.CALL else OptionType.CALL
        bond_opt = HW_ZCB_CallPutPrice(opt_type, Kk, lambd, eta, P0T, Tm, Tk)
        total += ck * bond_opt

    return N * total


def main():
    # Set the option type - change this to switch between CALL and PUT
    CP = OptionType.CALL  # Change to OptionType.PUT for Put option

    NoOfPaths = 20000
    NoOfSteps = 1000

    lambd = 0.02
    eta = 0.02

    P0T = lambda T: np.exp(-0.1 * T)  # np.exp(-0.03*T*T-0.1*T)
    r0 = HW_r_0(P0T, lambd, eta)

    N = 25
    T_end = 50
    Tgrid = np.linspace(0, T_end, N)
    Exact = np.zeros([N, 1])
    Proxy = np.zeros([N, 1])

    # ZCB validation plot
    for i, Ti in enumerate(Tgrid):
        Proxy[i] = HW_ZCB(lambd, eta, P0T, 0.0, Ti, r0)
        Exact[i] = P0T(Ti)

    # Plot ZCB comparison
    plt.figure(1)
    plt.grid()
    plt.plot(Tgrid, Exact, '-k')
    plt.plot(Tgrid, Proxy, '--r')
    plt.legend(["Analytical ZCB", "Monte Carlo ZCB"])
    plt.title('P(0,T) from Monte Carlo vs. Analytical expression')
    plt.xlabel('T')
    plt.ylabel('P(0,T)')
    plt.show()

    # Swaption parameters
    Tm = 5.0
    K_swap = 0.05
    payment_times = np.arange(6.0, 11.0, 1.0)
    accruals = [1.0] * len(payment_times)

    # Price swaption based on option type
    if CP == OptionType.CALL:
        price = price_swaption(OptionType.CALL, lambd, eta, P0T, Tm, payment_times, accruals, K_swap, N)
        print("Payer Swaption (Call) price:", price)

        # Plot price vs strike for call option
        strikes = np.linspace(0.01, 0.10, 10)
        prices_call = [price_swaption(OptionType.CALL, lambd, eta, P0T, Tm, payment_times, accruals, K, N) for K in
                       strikes]

        plt.figure(2)
        plt.plot(strikes, prices_call, marker='o', color='blue')
        plt.xlabel('Swap Rate Strike')
        plt.ylabel('Swaption Price')
        plt.title('Payer Swaption (Call) Prices under Hull-White Model')
        plt.grid(True)
        plt.show()

    elif CP == OptionType.PUT:
        price = price_swaption(OptionType.PUT, lambd, eta, P0T, Tm, payment_times, accruals, K_swap, N)
        print("Receiver Swaption (Put) price:", price)

        # Plot price vs strike for put option
        strikes = np.linspace(0.01, 0.10, 10)
        prices_put = [price_swaption(OptionType.PUT, lambd, eta, P0T, Tm, payment_times, accruals, K, N) for K in
                      strikes]

        plt.figure(2)
        plt.plot(strikes, prices_put, marker='x', color='red')
        plt.xlabel('Swap Rate Strike')
        plt.ylabel('Swaption Price')
        plt.title('Receiver Swaption (Put) Prices under Hull-White Model')
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    main()