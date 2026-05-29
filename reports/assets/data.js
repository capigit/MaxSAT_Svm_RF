window.LAD_REPORT_DATA = {
  "date_generation": "30/05/2026 01:08",
  "dataset": {
    "nom": "Breast Cancer Wisconsin binarisé",
    "nb_exemples": 569,
    "nb_train": 398,
    "nb_test": 171,
    "nb_variables": 30
  },
  "parametres": {
    "random_state": 42,
    "test_size": 0.3,
    "K": 10,
    "N_S": 15
  },
  "modeles": [
    {
      "nom": "SVM Classique",
      "famille": "SVM",
      "accuracy": 0.9707602339181286,
      "variables": 30,
      "lad": false
    },
    {
      "nom": "LAD-SVM",
      "famille": "SVM",
      "accuracy": 0.9766081871345029,
      "variables": 11,
      "lad": true
    },
    {
      "nom": "RF Classique",
      "famille": "Forêt aléatoire",
      "accuracy": 0.9532163742690059,
      "variables": 30,
      "lad": false
    },
    {
      "nom": "RF-LAD",
      "famille": "Forêt aléatoire",
      "accuracy": 0.9415204678362573,
      "variables": "~11.0",
      "lad": true
    }
  ],
  "mss_svm": [
    1,
    7,
    9,
    11,
    15,
    17,
    18,
    21,
    23,
    24,
    28
  ],
  "mss_rf": [
    [
      1,
      7,
      9,
      11,
      15,
      17,
      18,
      21,
      23,
      24,
      28
    ],
    [
      1,
      7,
      8,
      9,
      17,
      18,
      19,
      21,
      23,
      24,
      28
    ],
    [
      1,
      6,
      8,
      9,
      17,
      18,
      19,
      21,
      23,
      24,
      28
    ],
    [
      1,
      8,
      9,
      17,
      18,
      19,
      21,
      23,
      24,
      27,
      28
    ],
    [
      1,
      9,
      11,
      15,
      17,
      18,
      21,
      23,
      24,
      27,
      28
    ],
    [
      1,
      6,
      9,
      11,
      15,
      17,
      18,
      21,
      23,
      24,
      28
    ],
    [
      1,
      7,
      9,
      11,
      14,
      15,
      18,
      21,
      23,
      24,
      28
    ],
    [
      1,
      4,
      8,
      9,
      15,
      17,
      18,
      21,
      23,
      24,
      28
    ],
    [
      1,
      8,
      9,
      15,
      17,
      18,
      21,
      23,
      24,
      28,
      29
    ],
    [
      1,
      6,
      8,
      9,
      11,
      17,
      18,
      21,
      23,
      24,
      28
    ],
    [
      1,
      6,
      9,
      11,
      12,
      17,
      18,
      21,
      23,
      24,
      28
    ],
    [
      1,
      4,
      6,
      8,
      9,
      17,
      18,
      21,
      23,
      24,
      28
    ],
    [
      1,
      4,
      8,
      9,
      17,
      18,
      21,
      23,
      24,
      26,
      28
    ],
    [
      1,
      8,
      9,
      17,
      18,
      21,
      23,
      24,
      26,
      28,
      29
    ],
    [
      1,
      4,
      8,
      9,
      13,
      17,
      20,
      21,
      24,
      26,
      28
    ]
  ]
};
