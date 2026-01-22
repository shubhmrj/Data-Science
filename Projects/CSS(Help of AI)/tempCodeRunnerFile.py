import pytest
from unittesting import BankAccount

@pytest.fixture
def bank_account():
    return BankAccount(1000)  # Initial balance of 1000

def test_initial_balance():
    account = BankAccount(1000)
    assert account.balance == 1000

def test_negative_initial_balance():
    with pytest.raises(ValueError):
        BankAccount(-100)

def test_deposit(bank_account):
    bank_account.deposit(500)
    assert bank_account.balance == 1500

def test_withdraw(bank_account):
    bank_account.withdraw(500)
    assert bank_account.balance == 500

def test_multiple_transactions(bank_account):
    bank_account.deposit(500)  # 1500
    bank_account.withdraw(200) # 1300
    bank_account.deposit(100)  # 1400
    assert bank_account.balance == 1400

def test_withdraw_insufficient_funds(bank_account):
    with pytest.raises(ValueError):
        bank_account.withdraw(2000)

def test_negative_deposit(bank_account):
    with pytest.raises(ValueError):
        bank_account.deposit(-100)

def test_negative_withdrawal(bank_account):
    with pytest.raises(ValueError):
        bank_account.withdraw(-100)

def test_zero_deposit(bank_account):
    with pytest.raises(ValueError):
        bank_account.deposit(0)

def test_zero_withdrawal(bank_account):
    with pytest.raises(ValueError):
        bank_account.withdraw(0)

def test_str_representation():
    account = BankAccount(1234.56)
    assert str(account) == "Bank Account Balance: $1234.56"

def test_balance_type(bank_account):
    assert isinstance(bank_account.balance, (int, float))