# Dataset

The dataset is a collection of `.raw` files, containing the raw EEG and EMG signals for each mouse, and of `.smo` files,
which contain the annotated sleep stages for each mouse, as well as variability of the EEG and EMG signals and the
DFT power spectrum of the EEG signals.

Here's an overview of the structure you should have after downloading
the [dataset](https://drive.google.com/drive/folders/16skgaKif1u9WygjQ-a1StLtX-B-EfCid?usp=share_link) (requires an EPFL
account).

```text
├── data
│   ├── README.md                               # Instructions to obtain the full dataset
│   ├── smo                                     # SMO EEG and EMG signals
│   ├── csv                                     # Pre-processed CSV files
├── experiments                                 # ... rest of the project structure
```

You should download the contents of the `Reduced Dataset` folder, and place it directly in this folder.
