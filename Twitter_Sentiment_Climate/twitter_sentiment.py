"""
This is the main file which will be used to run the entire project.
"""
from src.i_fetch_data import (
    fetch_data
)
from src.ii_eda import (
    eda
)
from src.iii_preprocessing import (
    preprocessing
)
from src.v_data_segregation import (
    data_segregation
)
from src.vii_test import (
    test_model
)
from src.vi_train import (
    train
)
from src.iv_data_checks import (
    data_checks
)

if __name__ == "__main__":
    # Fetch the data
    fetch_data()

    # Perform EDA
    eda()

    # Perform preprocessing
    preprocessing()

    # Perform data checks
    data_checks()

    # Segregate the data
    data_segregation()

    # Train and test the model
    train()
    test_model()
