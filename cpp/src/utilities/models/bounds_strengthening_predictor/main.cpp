/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cmath>
#include "header.h"

#if defined(__clang__) || defined(__GNUC__)
#define LIKELY(x)   __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define LIKELY(x)   (x)
#define UNLIKELY(x) (x)
#endif
#define N_TARGET    1
#define MAX_N_CLASS 1

const unsigned char is_categorical[] = {
  0,
  0,
  0,
  0,
  0,
};
static const int32_t num_class[] = {
  1,
};

int32_t bounds_strengthening_predictor::get_num_target(void) { return std::exp(N_TARGET) - (100); }
void bounds_strengthening_predictor::get_num_class(int32_t* out)
{
  for (int i = 0; i < N_TARGET; ++i) {
    out[i] = num_class[i];
  }
}
int32_t bounds_strengthening_predictor::get_num_feature(void) { return std::exp(5) - (100); }
const char* bounds_strengthening_predictor::get_threshold_type(void) { return "float64"; }
const char* bounds_strengthening_predictor::get_leaf_output_type(void) { return "float64"; }

void bounds_strengthening_predictor::predict(union Entry* data, int pred_margin, double* result)
{
  // Quantize data
  for (int i = 0; i < 5; ++i) {
    if (data[i].missing != -1 && !is_categorical[i]) {
      data[i].qvalue = quantize(data[i].fvalue, i);
    }
  }

  unsigned int tmp;
  if (LIKELY(false || (data[3].qvalue <= 36))) {
    if (LIKELY(false || (data[3].qvalue <= 4))) {
      if (LIKELY(false || (data[0].qvalue <= 0))) {
        result[0] += 4.6207593335207156;
      } else {
        result[0] += 4.621034455402758;
      }
    } else {
      if (UNLIKELY(false || (data[3].qvalue <= 10))) {
        if (LIKELY(false || (data[0].qvalue <= 12))) {
          if (UNLIKELY(false || (data[3].qvalue <= 6))) {
            result[0] += 4.6219609086662095;
          } else {
            result[0] += 4.622183037068742;
          }
        } else {
          if (UNLIKELY(false || (data[0].qvalue <= 20))) {
            result[0] += 4.621033941004809;
          } else {
            if (LIKELY(false || (data[0].qvalue <= 26))) {
              result[0] += 4.621791267961554;
            } else {
              result[0] += 4.621426104022116;
            }
          }
        }
      } else {
        if (UNLIKELY(false || (data[1].qvalue <= 6))) {
          if (LIKELY(false || (data[3].qvalue <= 18))) {
            result[0] += 4.625919220945453;
          } else {
            result[0] += 4.628513174443221;
          }
        } else {
          if (UNLIKELY(false || (data[1].qvalue <= 26))) {
            if (UNLIKELY(false || (data[1].qvalue <= 10))) {
              result[0] += 4.621784796275526;
            } else {
              result[0] += 4.622332428952099;
            }
          } else {
            if (LIKELY(false || (data[0].qvalue <= 60))) {
              result[0] += 4.622921220395778;
            } else {
              result[0] += 4.625300570663533;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY(false || (data[1].qvalue <= 28))) {
      result[0] += 4.640761672090703;
    } else {
      if (LIKELY(false || (data[3].qvalue <= 56))) {
        if (UNLIKELY(false || (data[1].qvalue <= 34))) {
          if (UNLIKELY(false || (data[0].qvalue <= 24))) {
            result[0] += 4.6249068107234645;
          } else {
            result[0] += 4.622329447390833;
          }
        } else {
          if (LIKELY(false || (data[3].qvalue <= 48))) {
            if (LIKELY(false || (data[3].qvalue <= 44))) {
              result[0] += 4.629801464601299;
            } else {
              result[0] += 4.631153243632111;
            }
          } else {
            if (UNLIKELY(false || (data[3].qvalue <= 50))) {
              result[0] += 4.661969246306176;
            } else {
              result[0] += 4.634038239440175;
            }
          }
        }
      } else {
        if (UNLIKELY(false || (data[0].qvalue <= 32))) {
          result[0] += 4.661584263398344;
        } else {
          result[0] += 4.670140606548272;
        }
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 136))) {
    if (LIKELY(false || (data[2].qvalue <= 106))) {
      if (LIKELY(false || (data[1].qvalue <= 68))) {
        if (LIKELY(false || (data[2].qvalue <= 44))) {
          if (LIKELY(false || (data[0].qvalue <= 44))) {
            if (LIKELY(false || (data[0].qvalue <= 4))) {
              result[0] += -0.0015367070074457825;
            } else {
              result[0] += -0.0013136363962750828;
            }
          } else {
            result[0] += -4.5316595658010896e-05;
          }
        } else {
          if (LIKELY(false || (data[1].qvalue <= 26))) {
            if (LIKELY(false || (data[2].qvalue <= 94))) {
              result[0] += -0.0004993478758971943;
            } else {
              result[0] += 0.001360991802963148;
            }
          } else {
            if (LIKELY(false || (data[1].qvalue <= 48))) {
              result[0] += 0.0003523832199634608;
            } else {
              result[0] += 0.00182867874633131;
            }
          }
        }
      } else {
        if (LIKELY(false || (data[1].qvalue <= 76))) {
          if (UNLIKELY(false || (data[2].qvalue <= 30))) {
            result[0] += 0.002593102353418926;
          } else {
            result[0] += 0.005728859642902398;
          }
        } else {
          result[0] += 0.008440219358081076;
        }
      }
    } else {
      if (UNLIKELY(false || (data[2].qvalue <= 116))) {
        if (LIKELY(false || (data[1].qvalue <= 54))) {
          result[0] += 0.002535670244021562;
        } else {
          if (LIKELY(false || (data[0].qvalue <= 78))) {
            result[0] += 0.006909437253271745;
          } else {
            result[0] += 0.02304069650359452;
          }
        }
      } else {
        if (LIKELY(false || (data[1].qvalue <= 56))) {
          if (UNLIKELY(false || (data[2].qvalue <= 120))) {
            result[0] += 0.004730323543486456;
          } else {
            result[0] += 0.0061965453575640326;
          }
        } else {
          result[0] += 0.013205196280244542;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[2].qvalue <= 148))) {
      if (LIKELY(false || (data[0].qvalue <= 78))) {
        if (LIKELY(false || (data[2].qvalue <= 144))) {
          if (UNLIKELY(false || (data[2].qvalue <= 140))) {
            if (UNLIKELY(false || (data[2].qvalue <= 138))) {
              result[0] += 0.00807777003523674;
            } else {
              result[0] += 0.010705611526849181;
            }
          } else {
            result[0] += 0.012723627762072781;
          }
        } else {
          if (LIKELY(false || (data[1].qvalue <= 70))) {
            if (UNLIKELY(false || (data[2].qvalue <= 146))) {
              result[0] += 0.015703792435409165;
            } else {
              result[0] += 0.020898327013036955;
            }
          } else {
            result[0] += 0.02921710257052448;
          }
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 140))) {
          result[0] += 0.03916204050259713;
        } else {
          result[0] += 0.060036659967154266;
        }
      }
    } else {
      if (LIKELY(false || (data[0].qvalue <= 32))) {
        if (UNLIKELY(false || (data[2].qvalue <= 150))) {
          result[0] += 0.02552313877403762;
        } else {
          result[0] += 0.03311200053743086;
        }
      } else {
        if (UNLIKELY(false || (data[0].qvalue <= 36))) {
          result[0] += 0.1013872077379908;
        } else {
          if (UNLIKELY(false || (data[2].qvalue <= 152))) {
            if (UNLIKELY(false || (data[1].qvalue <= 72))) {
              result[0] += 0.026368786688846874;
            } else {
              result[0] += 0.0433115207428156;
            }
          } else {
            if (UNLIKELY(false || (data[0].qvalue <= 38))) {
              result[0] += 0.029699057000023982;
            } else {
              result[0] += 0.06580137491588535;
            }
          }
        }
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 136))) {
    if (LIKELY(false || (data[2].qvalue <= 104))) {
      if (LIKELY(false || (data[1].qvalue <= 68))) {
        if (LIKELY(false || (data[2].qvalue <= 46))) {
          if (LIKELY(false || (data[1].qvalue <= 44))) {
            if (LIKELY(false || (data[2].qvalue <= 18))) {
              result[0] += -0.0013768933256704324;
            } else {
              result[0] += -0.0011262398007476182;
            }
          } else {
            result[0] += -0.00032975454434196595;
          }
        } else {
          if (LIKELY(false || (data[1].qvalue <= 26))) {
            if (LIKELY(false || (data[2].qvalue <= 60))) {
              result[0] += -0.0006276937122835658;
            } else {
              result[0] += -0.00018352024583244162;
            }
          } else {
            if (LIKELY(false || (data[1].qvalue <= 48))) {
              result[0] += 0.00028540160605117557;
            } else {
              result[0] += 0.0016549226422226974;
            }
          }
        }
      } else {
        if (LIKELY(false || (data[1].qvalue <= 76))) {
          if (UNLIKELY(false || (data[2].qvalue <= 32))) {
            result[0] += 0.0023646158978421383;
          } else {
            result[0] += 0.0051637399936750455;
          }
        } else {
          result[0] += 0.0075761109087648;
        }
      }
    } else {
      if (UNLIKELY(false || (data[2].qvalue <= 114))) {
        if (LIKELY(false || (data[1].qvalue <= 54))) {
          result[0] += 0.0019452567655093584;
        } else {
          if (LIKELY(false || (data[1].qvalue <= 76))) {
            result[0] += 0.005666648643122513;
          } else {
            result[0] += 0.01829621969461441;
          }
        }
      } else {
        if (LIKELY(false || (data[1].qvalue <= 56))) {
          if (UNLIKELY(false || (data[2].qvalue <= 120))) {
            result[0] += 0.003990524959078667;
          } else {
            result[0] += 0.0055769135986496665;
          }
        } else {
          result[0] += 0.011831853925204368;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[2].qvalue <= 148))) {
      if (LIKELY(false || (data[1].qvalue <= 76))) {
        if (LIKELY(false || (data[2].qvalue <= 144))) {
          if (UNLIKELY(false || (data[2].qvalue <= 140))) {
            if (UNLIKELY(false || (data[2].qvalue <= 138))) {
              result[0] += 0.007270356084799834;
            } else {
              result[0] += 0.009635304835365038;
            }
          } else {
            result[0] += 0.011451321149275655;
          }
        } else {
          if (LIKELY(false || (data[3].qvalue <= 52))) {
            if (LIKELY(false || (data[1].qvalue <= 70))) {
              result[0] += 0.017390748142516205;
            } else {
              result[0] += 0.02630676100748059;
            }
          } else {
            result[0] += 0.009110929474456986;
          }
        }
      } else {
        if (UNLIKELY(false || (data[2].qvalue <= 138))) {
          result[0] += 0.02801222951444861;
        } else {
          result[0] += 0.042642993176138254;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 44))) {
        if (UNLIKELY(false || (data[2].qvalue <= 150))) {
          result[0] += 0.022974970733987354;
        } else {
          result[0] += 0.029654697301807826;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 58))) {
          if (UNLIKELY(false || (data[2].qvalue <= 152))) {
            if (UNLIKELY(false || (data[1].qvalue <= 72))) {
              result[0] += 0.019513382427394393;
            } else {
              result[0] += 0.039000512019146324;
            }
          } else {
            if (UNLIKELY(false || (data[1].qvalue <= 64))) {
              result[0] += 0.03719209477305412;
            } else {
              result[0] += 0.059241237849631215;
            }
          }
        } else {
          result[0] += 0.09132090954478399;
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 36))) {
    if (LIKELY(false || (data[3].qvalue <= 4))) {
      if (LIKELY(false || (data[3].qvalue <= 0))) {
        result[0] += -0.00125673851948232;
      } else {
        result[0] += -0.0010632420050787612;
      }
    } else {
      if (UNLIKELY(false || (data[3].qvalue <= 10))) {
        if (LIKELY(false || (data[0].qvalue <= 12))) {
          result[0] += -0.00024632899498939104;
        } else {
          if (UNLIKELY(false || (data[0].qvalue <= 20))) {
            result[0] += -0.0010346225974928775;
          } else {
            result[0] += -0.0005449025485433642;
          }
        }
      } else {
        if (UNLIKELY(false || (data[0].qvalue <= 10))) {
          if (UNLIKELY(false || (data[1].qvalue <= 10))) {
            if (LIKELY(false || (data[1].qvalue <= 2))) {
              result[0] += 0.0025363360589947676;
            } else {
              result[0] += 0.004674142936566838;
            }
          } else {
            result[0] += 0.0015267965552685568;
          }
        } else {
          if (LIKELY(false || (data[0].qvalue <= 60))) {
            if (UNLIKELY(false || (data[1].qvalue <= 26))) {
              result[0] += -0.00027483790298776263;
            } else {
              result[0] += 0.0002972757256301511;
            }
          } else {
            if (UNLIKELY(false || (data[0].qvalue <= 62))) {
              result[0] += 0.0041428877040166215;
            } else {
              result[0] += 0.0011655918164156917;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY(false || (data[1].qvalue <= 28))) {
      result[0] += 0.01333787139964387;
    } else {
      if (LIKELY(false || (data[3].qvalue <= 56))) {
        if (UNLIKELY(false || (data[1].qvalue <= 34))) {
          if (UNLIKELY(false || (data[0].qvalue <= 24))) {
            result[0] += 0.0015462619991967958;
          } else {
            result[0] += -0.0001502519938051531;
          }
        } else {
          if (LIKELY(false || (data[3].qvalue <= 48))) {
            if (LIKELY(false || (data[0].qvalue <= 70))) {
              result[0] += 0.005627108368440385;
            } else {
              result[0] += 0.004311373641741069;
            }
          } else {
            if (UNLIKELY(false || (data[3].qvalue <= 50))) {
              result[0] += 0.029228057063747634;
            } else {
              result[0] += 0.008457099724201248;
            }
          }
        }
      } else {
        result[0] += 0.03349293840846523;
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 136))) {
    if (LIKELY(false || (data[2].qvalue <= 94))) {
      if (LIKELY(false || (data[1].qvalue <= 70))) {
        if (LIKELY(false || (data[2].qvalue <= 42))) {
          if (LIKELY(false || (data[0].qvalue <= 44))) {
            result[0] += -0.0010804696960751716;
          } else {
            result[0] += -7.891687944899684e-05;
          }
        } else {
          if (LIKELY(false || (data[1].qvalue <= 26))) {
            if (LIKELY(false || (data[2].qvalue <= 60))) {
              result[0] += -0.0005406252210800222;
            } else {
              result[0] += -0.00018499077307288617;
            }
          } else {
            if (LIKELY(false || (data[1].qvalue <= 52))) {
              result[0] += 9.144939740089358e-05;
            } else {
              result[0] += 0.0011602440613788997;
            }
          }
        }
      } else {
        if (LIKELY(false || (data[1].qvalue <= 76))) {
          if (UNLIKELY(false || (data[2].qvalue <= 30))) {
            result[0] += 0.0015989411665534034;
          } else {
            result[0] += 0.004244323292266292;
          }
        } else {
          result[0] += 0.005838572629904283;
        }
      }
    } else {
      if (UNLIKELY(false || (data[2].qvalue <= 114))) {
        if (LIKELY(false || (data[1].qvalue <= 52))) {
          result[0] += 0.0014231677129056524;
        } else {
          if (LIKELY(false || (data[0].qvalue <= 78))) {
            result[0] += 0.004423386593859449;
          } else {
            result[0] += 0.014837017284128172;
          }
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 56))) {
          if (UNLIKELY(false || (data[2].qvalue <= 122))) {
            if (LIKELY(false || (data[1].qvalue <= 46))) {
              result[0] += 0.0030916099442218725;
            } else {
              result[0] += 0.0051002350582270845;
            }
          } else {
            result[0] += 0.00452717826425259;
          }
        } else {
          result[0] += 0.011297589064915738;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[2].qvalue <= 148))) {
      if (LIKELY(false || (data[0].qvalue <= 78))) {
        if (LIKELY(false || (data[2].qvalue <= 144))) {
          if (UNLIKELY(false || (data[2].qvalue <= 138))) {
            if (LIKELY(false || (data[1].qvalue <= 38))) {
              result[0] += 0.005096665266400936;
            } else {
              result[0] += 0.00914843088656596;
            }
          } else {
            result[0] += 0.008873224907066796;
          }
        } else {
          if (LIKELY(false || (data[1].qvalue <= 70))) {
            if (UNLIKELY(false || (data[2].qvalue <= 146))) {
              result[0] += 0.011259071094333009;
            } else {
              result[0] += 0.0157599764415071;
            }
          } else {
            result[0] += 0.022078909557385205;
          }
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 140))) {
          result[0] += 0.030722327275153918;
        } else {
          result[0] += 0.04927028425037861;
        }
      }
    } else {
      if (LIKELY(false || (data[0].qvalue <= 32))) {
        if (UNLIKELY(false || (data[2].qvalue <= 150))) {
          result[0] += 0.019342689499895135;
        } else {
          result[0] += 0.02546834143335123;
        }
      } else {
        if (UNLIKELY(false || (data[2].qvalue <= 152))) {
          if (UNLIKELY(false || (data[1].qvalue <= 72))) {
            result[0] += 0.018988974551944173;
          } else {
            result[0] += 0.03220939183373785;
          }
        } else {
          if (UNLIKELY(false || (data[0].qvalue <= 36))) {
            result[0] += 0.08284748123076036;
          } else {
            if (UNLIKELY(false || (data[0].qvalue <= 38))) {
              result[0] += 0.023946994628225055;
            } else {
              result[0] += 0.05204241935061829;
            }
          }
        }
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 136))) {
    if (LIKELY(false || (data[2].qvalue <= 102))) {
      if (LIKELY(false || (data[0].qvalue <= 74))) {
        if (LIKELY(false || (data[2].qvalue <= 48))) {
          if (LIKELY(false || (data[0].qvalue <= 30))) {
            if (LIKELY(false || (data[2].qvalue <= 18))) {
              result[0] += -0.001013660458835288;
            } else {
              result[0] += -0.0008091843403196929;
            }
          } else {
            if (LIKELY(false || (data[3].qvalue <= 32))) {
              result[0] += -0.0003093376277910651;
            } else {
              result[0] += 0.0008342959644312034;
            }
          }
        } else {
          if (UNLIKELY(false || (data[2].qvalue <= 62))) {
            if (LIKELY(false || (data[3].qvalue <= 8))) {
              result[0] += -0.00025794934360278315;
            } else {
              result[0] += -0.0006182054733506302;
            }
          } else {
            if (LIKELY(false || (data[2].qvalue <= 94))) {
              result[0] += 3.786358371332886e-05;
            } else {
              result[0] += 0.0011058684623919252;
            }
          }
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 38))) {
          result[0] += 0.003947623029554656;
        } else {
          if (LIKELY(false || (data[2].qvalue <= 88))) {
            result[0] += 0.005537522805788919;
          } else {
            result[0] += 0.012393589503644558;
          }
        }
      }
    } else {
      if (UNLIKELY(false || (data[2].qvalue <= 118))) {
        if (LIKELY(false || (data[0].qvalue <= 74))) {
          if (LIKELY(false || (data[0].qvalue <= 48))) {
            if (LIKELY(false || (data[2].qvalue <= 114))) {
              result[0] += 0.0015414300409981696;
            } else {
              result[0] += 0.0025675242802216634;
            }
          } else {
            result[0] += 0.00484969263475652;
          }
        } else {
          result[0] += 0.016452803226945166;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 56))) {
          result[0] += 0.0039797931775412965;
        } else {
          result[0] += 0.011972692174657348;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[2].qvalue <= 148))) {
      if (LIKELY(false || (data[0].qvalue <= 78))) {
        if (LIKELY(false || (data[2].qvalue <= 144))) {
          if (UNLIKELY(false || (data[2].qvalue <= 138))) {
            if (UNLIKELY(false || (data[3].qvalue <= 36))) {
              result[0] += 0.008268968051726045;
            } else {
              result[0] += 0.004574969868580042;
            }
          } else {
            result[0] += 0.007985935393142055;
          }
        } else {
          if (LIKELY(false || (data[3].qvalue <= 56))) {
            if (LIKELY(false || (data[0].qvalue <= 30))) {
              result[0] += 0.012855216147807361;
            } else {
              result[0] += 0.018209631592896635;
            }
          } else {
            result[0] += 0.0011929211910353282;
          }
        }
      } else {
        if (UNLIKELY(false || (data[2].qvalue <= 138))) {
          result[0] += 0.0213851068444448;
        } else {
          result[0] += 0.03414051433662315;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 44))) {
        if (UNLIKELY(false || (data[2].qvalue <= 150))) {
          result[0] += 0.017427519954483866;
        } else {
          result[0] += 0.022839697665974004;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 58))) {
          if (UNLIKELY(false || (data[2].qvalue <= 152))) {
            result[0] += 0.027770178962905873;
          } else {
            if (UNLIKELY(false || (data[0].qvalue <= 38))) {
              result[0] += 0.027731845133727596;
            } else {
              result[0] += 0.04685399499752964;
            }
          }
        } else {
          result[0] += 0.07097678249608726;
        }
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 136))) {
    if (LIKELY(false || (data[2].qvalue <= 102))) {
      if (LIKELY(false || (data[0].qvalue <= 74))) {
        if (LIKELY(false || (data[2].qvalue <= 40))) {
          if (LIKELY(false || (data[0].qvalue <= 54))) {
            if (LIKELY(false || (data[2].qvalue <= 18))) {
              result[0] += -0.0009069312057087865;
            } else {
              result[0] += -0.0007209240090653536;
            }
          } else {
            result[0] += 9.266279868544054e-06;
          }
        } else {
          if (UNLIKELY(false || (data[2].qvalue <= 60))) {
            if (LIKELY(false || (data[0].qvalue <= 46))) {
              result[0] += -0.00045145656190525603;
            } else {
              result[0] += 0.000736445256361047;
            }
          } else {
            if (LIKELY(false || (data[0].qvalue <= 48))) {
              result[0] += 3.824640547813124e-05;
            } else {
              result[0] += 0.0019395439371273169;
            }
          }
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 78))) {
          result[0] += 0.0032973085317471182;
        } else {
          if (LIKELY(false || (data[2].qvalue <= 72))) {
            result[0] += 0.0046992006748856376;
          } else {
            result[0] += 0.010662218181265368;
          }
        }
      }
    } else {
      if (UNLIKELY(false || (data[2].qvalue <= 114))) {
        if (LIKELY(false || (data[0].qvalue <= 48))) {
          result[0] += 0.0013872983040295432;
        } else {
          if (LIKELY(false || (data[0].qvalue <= 78))) {
            result[0] += 0.003922717377135387;
          } else {
            result[0] += 0.012402207983552287;
          }
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 56))) {
          if (UNLIKELY(false || (data[2].qvalue <= 122))) {
            result[0] += 0.0027351189679491643;
          } else {
            result[0] += 0.003676515596417783;
          }
        } else {
          if (LIKELY(false || (data[0].qvalue <= 78))) {
            result[0] += 0.008212890325666373;
          } else {
            result[0] += 0.018472826568917796;
          }
        }
      }
    }
  } else {
    if (LIKELY(false || (data[2].qvalue <= 148))) {
      if (LIKELY(false || (data[0].qvalue <= 78))) {
        if (LIKELY(false || (data[2].qvalue <= 144))) {
          if (UNLIKELY(false || (data[2].qvalue <= 138))) {
            if (LIKELY(false || (data[0].qvalue <= 36))) {
              result[0] += 0.004094328568438451;
            } else {
              result[0] += 0.007676761387855601;
            }
          } else {
            result[0] += 0.00718737123039425;
          }
        } else {
          if (LIKELY(false || (data[3].qvalue <= 56))) {
            if (LIKELY(false || (data[0].qvalue <= 30))) {
              result[0] += 0.011570012574530879;
            } else {
              result[0] += 0.016393931404272945;
            }
          } else {
            result[0] += 0.0010741865297096182;
          }
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 140))) {
          result[0] += 0.024740276188422474;
        } else {
          result[0] += 0.041189863514155156;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 44))) {
        if (UNLIKELY(false || (data[2].qvalue <= 150))) {
          result[0] += 0.01568551045560334;
        } else {
          result[0] += 0.020556299621416432;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 58))) {
          if (UNLIKELY(false || (data[2].qvalue <= 152))) {
            result[0] += 0.025004828468340786;
          } else {
            if (UNLIKELY(false || (data[0].qvalue <= 38))) {
              result[0] += 0.02500338864662955;
            } else {
              result[0] += 0.0421828362800883;
            }
          }
        } else {
          result[0] += 0.06392980175664915;
        }
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 136))) {
    if (LIKELY(false || (data[2].qvalue <= 94))) {
      if (LIKELY(false || (data[0].qvalue <= 74))) {
        if (LIKELY(false || (data[2].qvalue <= 40))) {
          if (LIKELY(false || (data[0].qvalue <= 30))) {
            result[0] += -0.0007939024435866282;
          } else {
            if (LIKELY(false || (data[3].qvalue <= 32))) {
              result[0] += -0.00028064729815836673;
            } else {
              result[0] += 0.0007573829970680826;
            }
          }
        } else {
          if (LIKELY(false || (data[0].qvalue <= 46))) {
            if (LIKELY(false || (data[3].qvalue <= 46))) {
              result[0] += -0.0002029265825135938;
            } else {
              result[0] += 0.0049281532625337285;
            }
          } else {
            if (LIKELY(false || (data[2].qvalue <= 66))) {
              result[0] += 0.0006539904593123779;
            } else {
              result[0] += 0.002101938298715612;
            }
          }
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 78))) {
          result[0] += 0.002966716939082758;
        } else {
          result[0] += 0.004282679307075282;
        }
      }
    } else {
      if (UNLIKELY(false || (data[2].qvalue <= 114))) {
        if (LIKELY(false || (data[0].qvalue <= 48))) {
          result[0] += 0.0010976994464251622;
        } else {
          if (LIKELY(false || (data[0].qvalue <= 78))) {
            result[0] += 0.003309816493639741;
          } else {
            result[0] += 0.010820436719805003;
          }
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 56))) {
          if (UNLIKELY(false || (data[2].qvalue <= 118))) {
            result[0] += 0.0020919978318753476;
          } else {
            if (UNLIKELY(false || (data[3].qvalue <= 28))) {
              result[0] += 0.005176501864572617;
            } else {
              result[0] += 0.0031809497994157152;
            }
          }
        } else {
          if (LIKELY(false || (data[0].qvalue <= 78))) {
            result[0] += 0.007393664821488642;
          } else {
            result[0] += 0.016667527793483305;
          }
        }
      }
    }
  } else {
    if (LIKELY(false || (data[2].qvalue <= 148))) {
      if (LIKELY(false || (data[0].qvalue <= 80))) {
        if (LIKELY(false || (data[2].qvalue <= 144))) {
          if (UNLIKELY(false || (data[2].qvalue <= 140))) {
            if (LIKELY(false || (data[0].qvalue <= 78))) {
              result[0] += 0.005229670373125717;
            } else {
              result[0] += 0.015644584188655934;
            }
          } else {
            result[0] += 0.006607172961087171;
          }
        } else {
          if (LIKELY(false || (data[3].qvalue <= 56))) {
            if (LIKELY(false || (data[0].qvalue <= 30))) {
              result[0] += 0.010413297684558929;
            } else {
              result[0] += 0.014759276311661552;
            }
          } else {
            result[0] += 0.0009672698351743883;
          }
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 140))) {
          result[0] += 0.024270801343479934;
        } else {
          result[0] += 0.03719959639012814;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 44))) {
        if (UNLIKELY(false || (data[2].qvalue <= 150))) {
          result[0] += 0.014117627727620947;
        } else {
          result[0] += 0.018501183876499263;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 58))) {
          if (UNLIKELY(false || (data[2].qvalue <= 152))) {
            result[0] += 0.022514852274741447;
          } else {
            if (UNLIKELY(false || (data[0].qvalue <= 38))) {
              result[0] += 0.022543378573271537;
            } else {
              result[0] += 0.03797737476673532;
            }
          }
        } else {
          result[0] += 0.057582486043418105;
        }
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 136))) {
    if (LIKELY(false || (data[2].qvalue <= 94))) {
      if (LIKELY(false || (data[0].qvalue <= 74))) {
        if (LIKELY(false || (data[2].qvalue <= 28))) {
          if (LIKELY(false || (data[0].qvalue <= 54))) {
            result[0] += -0.0007216230613638112;
          } else {
            result[0] += 1.2243669808756593e-05;
          }
        } else {
          if (UNLIKELY(false || (data[2].qvalue <= 60))) {
            if (LIKELY(false || (data[0].qvalue <= 30))) {
              result[0] += -0.00040489502028823784;
            } else {
              result[0] += 0.00042280612429353794;
            }
          } else {
            if (LIKELY(false || (data[3].qvalue <= 46))) {
              result[0] += 6.1268474691525876e-06;
            } else {
              result[0] += 0.004594043928912372;
            }
          }
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 38))) {
          result[0] += 0.002846796010984943;
        } else {
          result[0] += 0.004154787212310116;
        }
      }
    } else {
      if (UNLIKELY(false || (data[2].qvalue <= 112))) {
        if (LIKELY(false || (data[0].qvalue <= 76))) {
          if (LIKELY(false || (data[0].qvalue <= 48))) {
            result[0] += 0.0009445975360671696;
          } else {
            result[0] += 0.0025639599906144878;
          }
        } else {
          result[0] += 0.008383097287811591;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 56))) {
          if (UNLIKELY(false || (data[2].qvalue <= 118))) {
            result[0] += 0.0017724753898680124;
          } else {
            if (UNLIKELY(false || (data[3].qvalue <= 28))) {
              result[0] += 0.004659576685877428;
            } else {
              result[0] += 0.002862865819246229;
            }
          }
        } else {
          if (LIKELY(false || (data[0].qvalue <= 78))) {
            if (LIKELY(false || (data[2].qvalue <= 120))) {
              result[0] += 0.0046025811935695195;
            } else {
              result[0] += 0.010899977739561688;
            }
          } else {
            result[0] += 0.014507038097267284;
          }
        }
      }
    }
  } else {
    if (LIKELY(false || (data[2].qvalue <= 148))) {
      if (LIKELY(false || (data[0].qvalue <= 80))) {
        if (LIKELY(false || (data[2].qvalue <= 144))) {
          if (UNLIKELY(false || (data[2].qvalue <= 138))) {
            if (LIKELY(false || (data[0].qvalue <= 36))) {
              result[0] += 0.003162446156626471;
            } else {
              result[0] += 0.007225625842537261;
            }
          } else {
            result[0] += 0.005829436446592702;
          }
        } else {
          if (LIKELY(false || (data[3].qvalue <= 56))) {
            if (UNLIKELY(false || (data[2].qvalue <= 146))) {
              result[0] += 0.008519365467784451;
            } else {
              result[0] += 0.010728541856931856;
            }
          } else {
            result[0] += 0.0008709948831324892;
          }
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 140))) {
          result[0] += 0.021860231652754504;
        } else {
          result[0] += 0.0335958852712065;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 44))) {
        if (UNLIKELY(false || (data[2].qvalue <= 150))) {
          result[0] += 0.012706466408332943;
        } else {
          result[0] += 0.01665152863778353;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 58))) {
          if (UNLIKELY(false || (data[2].qvalue <= 152))) {
            result[0] += 0.02027282733611074;
          } else {
            if (UNLIKELY(false || (data[0].qvalue <= 38))) {
              result[0] += 0.02032540112733841;
            } else {
              result[0] += 0.03419118077768789;
            }
          }
        } else {
          result[0] += 0.051865370802847414;
        }
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 136))) {
    if (LIKELY(false || (data[2].qvalue <= 94))) {
      if (LIKELY(false || (data[0].qvalue <= 74))) {
        if (LIKELY(false || (data[2].qvalue <= 48))) {
          if (LIKELY(false || (data[0].qvalue <= 30))) {
            if (LIKELY(false || (data[2].qvalue <= 16))) {
              result[0] += -0.0006709370987019684;
            } else {
              result[0] += -0.0005270795525272784;
            }
          } else {
            if (LIKELY(false || (data[3].qvalue <= 32))) {
              result[0] += -0.00019736253512332336;
            } else {
              result[0] += 0.0006554788837920564;
            }
          }
        } else {
          if (LIKELY(false || (data[0].qvalue <= 46))) {
            if (LIKELY(false || (data[3].qvalue <= 46))) {
              result[0] += -0.00014697141542902774;
            } else {
              result[0] += 0.004114945413306209;
            }
          } else {
            if (LIKELY(false || (data[2].qvalue <= 66))) {
              result[0] += 0.000608774604580438;
            } else {
              result[0] += 0.0018912343738969528;
            }
          }
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 78))) {
          result[0] += 0.0023794517398263364;
        } else {
          result[0] += 0.0034944486152153164;
        }
      }
    } else {
      if (UNLIKELY(false || (data[2].qvalue <= 112))) {
        if (LIKELY(false || (data[0].qvalue <= 76))) {
          result[0] += 0.0009107471036068726;
        } else {
          result[0] += 0.007554206584444208;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 52))) {
          if (UNLIKELY(false || (data[2].qvalue <= 122))) {
            result[0] += 0.0018167266037088525;
          } else {
            result[0] += 0.002703491325573408;
          }
        } else {
          if (LIKELY(false || (data[0].qvalue <= 78))) {
            if (LIKELY(false || (data[2].qvalue <= 120))) {
              result[0] += 0.004131123450839368;
            } else {
              result[0] += 0.009561926307650204;
            }
          } else {
            result[0] += 0.013076206701871469;
          }
        }
      }
    }
  } else {
    if (LIKELY(false || (data[2].qvalue <= 148))) {
      if (LIKELY(false || (data[0].qvalue <= 78))) {
        if (LIKELY(false || (data[2].qvalue <= 144))) {
          if (UNLIKELY(false || (data[2].qvalue <= 138))) {
            if (UNLIKELY(false || (data[3].qvalue <= 36))) {
              result[0] += 0.005894735890468045;
            } else {
              result[0] += 0.0027635595215004857;
            }
          } else {
            result[0] += 0.005246514394410695;
          }
        } else {
          if (LIKELY(false || (data[3].qvalue <= 56))) {
            if (LIKELY(false || (data[0].qvalue <= 30))) {
              result[0] += 0.008396656550227073;
            } else {
              result[0] += 0.012412600592852905;
            }
          } else {
            result[0] += 0.0007843023806235918;
          }
        }
      } else {
        if (UNLIKELY(false || (data[2].qvalue <= 138))) {
          result[0] += 0.013815322846990745;
        } else {
          result[0] += 0.022879901539002146;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 44))) {
        if (UNLIKELY(false || (data[2].qvalue <= 150))) {
          result[0] += 0.011436360873006497;
        } else {
          result[0] += 0.014986792630805573;
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 58))) {
          if (UNLIKELY(false || (data[2].qvalue <= 152))) {
            result[0] += 0.018254062129975056;
          } else {
            if (UNLIKELY(false || (data[0].qvalue <= 38))) {
              result[0] += 0.018325642903990143;
            } else {
              result[0] += 0.03078245439443534;
            }
          }
        } else {
          result[0] += 0.04671587835039412;
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 36))) {
    if (LIKELY(false || (data[3].qvalue <= 2))) {
      if (LIKELY(false || (data[3].qvalue <= 0))) {
        result[0] += -0.0006125589231343043;
      } else {
        result[0] += -0.0004916423019606601;
      }
    } else {
      if (UNLIKELY(false || (data[0].qvalue <= 8))) {
        if (UNLIKELY(false || (data[1].qvalue <= 10))) {
          if (LIKELY(false || (data[1].qvalue <= 2))) {
            result[0] += 0.001216506849361631;
          } else {
            result[0] += 0.00271299155013397;
          }
        } else {
          result[0] += 0.0008440598879826304;
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 34))) {
          if (LIKELY(false || (data[0].qvalue <= 14))) {
            if (LIKELY(false || (data[3].qvalue <= 10))) {
              result[0] += -0.00012183459142403321;
            } else {
              result[0] += 0.00029635306618271324;
            }
          } else {
            if (UNLIKELY(false || (data[0].qvalue <= 22))) {
              result[0] += -0.00047278908965180035;
            } else {
              result[0] += -0.00026271782481790195;
            }
          }
        } else {
          if (LIKELY(false || (data[3].qvalue <= 26))) {
            if (LIKELY(false || (data[0].qvalue <= 60))) {
              result[0] += 0.00016567086050993653;
            } else {
              result[0] += 0.0012740789559656865;
            }
          } else {
            result[0] += 0.00119403700340191;
          }
        }
      }
    }
  } else {
    if (UNLIKELY(false || (data[1].qvalue <= 28))) {
      result[0] += 0.006378599209839731;
    } else {
      if (LIKELY(false || (data[3].qvalue <= 56))) {
        if (UNLIKELY(false || (data[1].qvalue <= 34))) {
          result[0] += -0.00013731745398177464;
        } else {
          if (LIKELY(false || (data[3].qvalue <= 48))) {
            if (LIKELY(false || (data[3].qvalue <= 44))) {
              result[0] += 0.0024863318876363182;
            } else {
              result[0] += 0.003201281592347354;
            }
          } else {
            if (UNLIKELY(false || (data[3].qvalue <= 50))) {
              result[0] += 0.014536959755708252;
            } else {
              result[0] += 0.004109250766509175;
            }
          }
        }
      } else {
        result[0] += 0.0156613343672559;
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 136))) {
    if (LIKELY(false || (data[2].qvalue <= 94))) {
      if (LIKELY(false || (data[1].qvalue <= 66))) {
        if (LIKELY(false || (data[2].qvalue <= 58))) {
          if (LIKELY(false || (data[1].qvalue <= 22))) {
            result[0] += -0.0005258393188517168;
          } else {
            if (LIKELY(false || (data[2].qvalue <= 54))) {
              result[0] += -0.00033817430900207735;
            } else {
              result[0] += 1.2707417395591901e-05;
            }
          }
        } else {
          if (LIKELY(false || (data[1].qvalue <= 52))) {
            if (UNLIKELY(false || (data[1].qvalue <= 8))) {
              result[0] += -0.0004450469071967257;
            } else {
              result[0] += 5.459686661712705e-05;
            }
          } else {
            result[0] += 0.0010611375199245895;
          }
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 36))) {
          result[0] += 0.0014671398186970237;
        } else {
          result[0] += 0.002626165800361098;
        }
      }
    } else {
      if (UNLIKELY(false || (data[2].qvalue <= 112))) {
        if (LIKELY(false || (data[1].qvalue <= 52))) {
          result[0] += 0.0007307625021520472;
        } else {
          result[0] += 0.0025431188996113325;
        }
      } else {
        if (LIKELY(false || (data[1].qvalue <= 54))) {
          if (UNLIKELY(false || (data[2].qvalue <= 122))) {
            if (LIKELY(false || (data[1].qvalue <= 40))) {
              result[0] += 0.0011336443912753236;
            } else {
              result[0] += 0.0026154250468263807;
            }
          } else {
            result[0] += 0.0021727864966215654;
          }
        } else {
          if (LIKELY(false || (data[0].qvalue <= 78))) {
            if (LIKELY(false || (data[2].qvalue <= 120))) {
              result[0] += 0.0036860385022813268;
            } else {
              result[0] += 0.0074261350156955945;
            }
          } else {
            result[0] += 0.011381202510825986;
          }
        }
      }
    }
  } else {
    if (LIKELY(false || (data[2].qvalue <= 148))) {
      if (LIKELY(false || (data[0].qvalue <= 78))) {
        if (LIKELY(false || (data[2].qvalue <= 146))) {
          if (LIKELY(false || (data[1].qvalue <= 70))) {
            if (UNLIKELY(false || (data[2].qvalue <= 138))) {
              result[0] += 0.0025260287423364753;
            } else {
              result[0] += 0.004205124307871224;
            }
          } else {
            if (UNLIKELY(false || (data[0].qvalue <= 38))) {
              result[0] += 0.01428713906478215;
            } else {
              result[0] += 0.006449499560970644;
            }
          }
        } else {
          result[0] += 0.008158350036622746;
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 140))) {
          result[0] += 0.015911680867656684;
        } else {
          result[0] += 0.027726709544658664;
        }
      }
    } else {
      if (LIKELY(false || (data[0].qvalue <= 32))) {
        if (UNLIKELY(false || (data[2].qvalue <= 150))) {
          result[0] += 0.009635705460455004;
        } else {
          if (LIKELY(false || (data[2].qvalue <= 152))) {
            result[0] += 0.012077164092969977;
          } else {
            result[0] += 0.013736665018938832;
          }
        }
      } else {
        if (UNLIKELY(false || (data[2].qvalue <= 152))) {
          if (UNLIKELY(false || (data[2].qvalue <= 150))) {
            result[0] += 0.0021803621484432372;
          } else {
            result[0] += 0.015298270954553467;
          }
        } else {
          if (UNLIKELY(false || (data[0].qvalue <= 36))) {
            result[0] += 0.04418030004892776;
          } else {
            if (UNLIKELY(false || (data[0].qvalue <= 38))) {
              result[0] += 0.012503185447837626;
            } else {
              result[0] += 0.02704296985360522;
            }
          }
        }
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 136))) {
    if (LIKELY(false || (data[2].qvalue <= 94))) {
      if (LIKELY(false || (data[1].qvalue <= 66))) {
        if (LIKELY(false || (data[2].qvalue <= 58))) {
          if (LIKELY(false || (data[1].qvalue <= 22))) {
            if (LIKELY(false || (data[2].qvalue <= 12))) {
              result[0] += -0.0005057327530247406;
            } else {
              result[0] += -0.0004075698534556644;
            }
          } else {
            if (UNLIKELY(false || (data[2].qvalue <= 14))) {
              result[0] += -0.0003907346454170061;
            } else {
              result[0] += -0.00013304885629976585;
            }
          }
        } else {
          if (LIKELY(false || (data[1].qvalue <= 52))) {
            if (UNLIKELY(false || (data[1].qvalue <= 8))) {
              result[0] += -0.000400544289864025;
            } else {
              result[0] += 4.913722296179108e-05;
            }
          } else {
            result[0] += 0.0009550458929996977;
          }
        }
      } else {
        if (LIKELY(false || (data[1].qvalue <= 76))) {
          if (UNLIKELY(false || (data[2].qvalue <= 34))) {
            result[0] += 0.0007601159979557645;
          } else {
            result[0] += 0.0019476041107708356;
          }
        } else {
          result[0] += 0.0025254091774788738;
        }
      }
    } else {
      if (UNLIKELY(false || (data[2].qvalue <= 116))) {
        if (LIKELY(false || (data[1].qvalue <= 52))) {
          result[0] += 0.000687914586801239;
        } else {
          if (LIKELY(false || (data[0].qvalue <= 78))) {
            result[0] += 0.0024117269694104632;
          } else {
            result[0] += 0.007822948252922929;
          }
        }
      } else {
        if (LIKELY(false || (data[1].qvalue <= 50))) {
          result[0] += 0.0018173927952783312;
        } else {
          result[0] += 0.005066485649762486;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[2].qvalue <= 148))) {
      if (LIKELY(false || (data[0].qvalue <= 78))) {
        if (LIKELY(false || (data[2].qvalue <= 146))) {
          if (LIKELY(false || (data[1].qvalue <= 70))) {
            if (UNLIKELY(false || (data[2].qvalue <= 138))) {
              result[0] += 0.002273550315729467;
            } else {
              result[0] += 0.0037846264364073015;
            }
          } else {
            if (UNLIKELY(false || (data[0].qvalue <= 38))) {
              result[0] += 0.012868415887070285;
            } else {
              result[0] += 0.005806636735887348;
            }
          }
        } else {
          result[0] += 0.007342867370677843;
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 140))) {
          result[0] += 0.014328672563322842;
        } else {
          result[0] += 0.025040684528648854;
        }
      }
    } else {
      if (LIKELY(false || (data[0].qvalue <= 32))) {
        if (UNLIKELY(false || (data[2].qvalue <= 150))) {
          result[0] += 0.008672550231782946;
        } else {
          if (LIKELY(false || (data[2].qvalue <= 152))) {
            result[0] += 0.010870030198642126;
          } else {
            result[0] += 0.012363709845251715;
          }
        }
      } else {
        if (UNLIKELY(false || (data[2].qvalue <= 152))) {
          if (UNLIKELY(false || (data[0].qvalue <= 38))) {
            if (UNLIKELY(false || (data[2].qvalue <= 150))) {
              result[0] += -0.0016214965620348532;
            } else {
              result[0] += 0.009768780560755148;
            }
          } else {
            result[0] += 0.014496867406714756;
          }
        } else {
          if (UNLIKELY(false || (data[0].qvalue <= 36))) {
            result[0] += 0.03979524121133249;
          } else {
            if (UNLIKELY(false || (data[0].qvalue <= 38))) {
              result[0] += 0.011297521186726434;
            } else {
              result[0] += 0.024346893042280684;
            }
          }
        }
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 136))) {
    if (LIKELY(false || (data[2].qvalue <= 96))) {
      if (LIKELY(false || (data[0].qvalue <= 74))) {
        if (LIKELY(false || (data[2].qvalue <= 58))) {
          if (LIKELY(false || (data[2].qvalue <= 24))) {
            if (LIKELY(false || (data[0].qvalue <= 54))) {
              result[0] += -0.0004359996480215986;
            } else {
              result[0] += 0;
            }
          } else {
            if (LIKELY(false || (data[0].qvalue <= 40))) {
              result[0] += -0.0002523395445968973;
            } else {
              result[0] += 0.0003694161280901895;
            }
          }
        } else {
          if (LIKELY(false || (data[3].qvalue <= 12))) {
            if (LIKELY(false || (data[2].qvalue <= 78))) {
              result[0] += 7.525930391762016e-06;
            } else {
              result[0] += 0.0008929398582163592;
            }
          } else {
            if (LIKELY(false || (data[0].qvalue <= 48))) {
              result[0] += -0.0001610122359790055;
            } else {
              result[0] += 0.0010397722343388125;
            }
          }
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 52))) {
          if (UNLIKELY(false || (data[2].qvalue <= 4))) {
            result[0] += -0.0005520984639497772;
          } else {
            if (LIKELY(false || (data[0].qvalue <= 78))) {
              result[0] += 0.0014389645609665816;
            } else {
              result[0] += 0.0022859821912433556;
            }
          }
        } else {
          result[0] += 0.004341192593591669;
        }
      }
    } else {
      if (UNLIKELY(false || (data[2].qvalue <= 118))) {
        if (LIKELY(false || (data[0].qvalue <= 74))) {
          if (LIKELY(false || (data[0].qvalue <= 48))) {
            if (LIKELY(false || (data[3].qvalue <= 42))) {
              result[0] += 0.0007445782254113342;
            } else {
              result[0] += -0.0004863660791352827;
            }
          } else {
            result[0] += 0.0022185971404081856;
          }
        } else {
          result[0] += 0.00764254255032432;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 28))) {
          result[0] += 0.003993263532652023;
        } else {
          result[0] += 0.0016802334304175982;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[2].qvalue <= 148))) {
      if (LIKELY(false || (data[0].qvalue <= 80))) {
        if (LIKELY(false || (data[2].qvalue <= 146))) {
          if (LIKELY(false || (data[3].qvalue <= 54))) {
            if (LIKELY(false || (data[2].qvalue <= 142))) {
              result[0] += 0.0032242882960233418;
            } else {
              result[0] += 0.004681685073819564;
            }
          } else {
            result[0] += -0.0019765970439377645;
          }
        } else {
          result[0] += 0.006608897421772834;
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 140))) {
          result[0] += 0.01415305377345304;
        } else {
          result[0] += 0.02261486836243421;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 44))) {
        if (UNLIKELY(false || (data[2].qvalue <= 150))) {
          result[0] += 0.007850465295499818;
        } else {
          result[0] += 0.0104151568660406;
        }
      } else {
        if (UNLIKELY(false || (data[2].qvalue <= 152))) {
          if (UNLIKELY(false || (data[0].qvalue <= 38))) {
            result[0] += -0.00899228106430244;
          } else {
            result[0] += 0.012754785278791262;
          }
        } else {
          if (LIKELY(false || (data[3].qvalue <= 58))) {
            if (UNLIKELY(false || (data[0].qvalue <= 38))) {
              result[0] += 0.01241382271659771;
            } else {
              result[0] += 0.02191960403899695;
            }
          } else {
            result[0] += 0.035845413210231866;
          }
        }
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 138))) {
    if (LIKELY(false || (data[2].qvalue <= 104))) {
      if (LIKELY(false || (data[1].qvalue <= 66))) {
        if (LIKELY(false || (data[2].qvalue <= 60))) {
          if (LIKELY(false || (data[1].qvalue <= 22))) {
            if (LIKELY(false || (data[3].qvalue <= 38))) {
              result[0] += -0.00038117538124280044;
            } else {
              result[0] += -0.0044417730590026435;
            }
          } else {
            if (LIKELY(false || (data[2].qvalue <= 54))) {
              result[0] += -0.00024944182998305845;
            } else {
              result[0] += 4.7726306723714356e-05;
            }
          }
        } else {
          if (LIKELY(false || (data[1].qvalue <= 48))) {
            if (UNLIKELY(false || (data[3].qvalue <= 12))) {
              result[0] += 0.00017780524533208728;
            } else {
              result[0] += -0.00014198949040707064;
            }
          } else {
            result[0] += 0.0008690091555703179;
          }
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 44))) {
          if (UNLIKELY(false || (data[2].qvalue <= 6))) {
            result[0] += -4.2794447077009146e-05;
          } else {
            result[0] += 0.0013736436855736134;
          }
        } else {
          result[0] += 0.0023003491149758966;
        }
      }
    } else {
      if (LIKELY(false || (data[1].qvalue <= 48))) {
        if (UNLIKELY(false || (data[2].qvalue <= 122))) {
          if (LIKELY(false || (data[3].qvalue <= 38))) {
            result[0] += 0.0008322321038850688;
          } else {
            result[0] += -0.0007022737671788451;
          }
        } else {
          if (UNLIKELY(false || (data[3].qvalue <= 36))) {
            result[0] += 0.0040552217853155985;
          } else {
            result[0] += 0.001568199690420882;
          }
        }
      } else {
        if (LIKELY(false || (data[1].qvalue <= 74))) {
          if (LIKELY(false || (data[2].qvalue <= 118))) {
            result[0] += 0.002158096393788496;
          } else {
            result[0] += 0.00418981399519502;
          }
        } else {
          result[0] += 0.007976203340311152;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[2].qvalue <= 148))) {
      if (LIKELY(false || (data[1].qvalue <= 70))) {
        if (LIKELY(false || (data[2].qvalue <= 146))) {
          if (LIKELY(false || (data[3].qvalue <= 54))) {
            if (LIKELY(false || (data[2].qvalue <= 142))) {
              result[0] += 0.0029787778286427856;
            } else {
              result[0] += 0.00387467999983618;
            }
          } else {
            result[0] += -0.0016447041228178653;
          }
        } else {
          result[0] += 0.005939025993708955;
        }
      } else {
        if (LIKELY(false || (data[1].qvalue <= 78))) {
          result[0] += 0.009542470806448608;
        } else {
          result[0] += 0.015164196475275924;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 44))) {
        if (UNLIKELY(false || (data[2].qvalue <= 150))) {
          result[0] += 0.0070657533568465065;
        } else {
          if (LIKELY(false || (data[2].qvalue <= 152))) {
            result[0] += 0.008770727747783912;
          } else {
            result[0] += 0.010031409549326995;
          }
        }
      } else {
        if (UNLIKELY(false || (data[2].qvalue <= 152))) {
          if (UNLIKELY(false || (data[1].qvalue <= 72))) {
            result[0] += -0.005887114820381005;
          } else {
            result[0] += 0.011992559542402971;
          }
        } else {
          if (LIKELY(false || (data[3].qvalue <= 58))) {
            if (UNLIKELY(false || (data[1].qvalue <= 64))) {
              result[0] += 0.011192463174135607;
            } else {
              result[0] += 0.019734306209876366;
            }
          } else {
            result[0] += 0.032287623295143474;
          }
        }
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 138))) {
    if (LIKELY(false || (data[2].qvalue <= 98))) {
      if (LIKELY(false || (data[0].qvalue <= 74))) {
        if (LIKELY(false || (data[2].qvalue <= 62))) {
          if (LIKELY(false || (data[2].qvalue <= 22))) {
            if (LIKELY(false || (data[3].qvalue <= 34))) {
              result[0] += -0.00034750254877484433;
            } else {
              result[0] += -0.0009661001233083217;
            }
          } else {
            if (LIKELY(false || (data[0].qvalue <= 46))) {
              result[0] += -0.0002100971251621207;
            } else {
              result[0] += 0.0003227294420283331;
            }
          }
        } else {
          if (UNLIKELY(false || (data[3].qvalue <= 12))) {
            if (LIKELY(false || (data[2].qvalue <= 80))) {
              result[0] += 6.630874663410788e-05;
            } else {
              result[0] += 0.0007797657979206323;
            }
          } else {
            if (LIKELY(false || (data[0].qvalue <= 48))) {
              result[0] += -0.00012124022680195402;
            } else {
              result[0] += 0.0009324062666283081;
            }
          }
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 52))) {
          result[0] += 0.0014435021333467982;
        } else {
          result[0] += 0.003681684849152054;
        }
      }
    } else {
      if (UNLIKELY(false || (data[2].qvalue <= 118))) {
        if (LIKELY(false || (data[3].qvalue <= 46))) {
          if (LIKELY(false || (data[0].qvalue <= 48))) {
            if (LIKELY(false || (data[3].qvalue <= 42))) {
              result[0] += 0.0006546173228667891;
            } else {
              result[0] += -0.0008103936292200276;
            }
          } else {
            if (UNLIKELY(false || (data[0].qvalue <= 50))) {
              result[0] += 0.006212216600054695;
            } else {
              result[0] += 0.001728401157423842;
            }
          }
        } else {
          result[0] += 0.005068273878703855;
        }
      } else {
        if (UNLIKELY(false || (data[3].qvalue <= 36))) {
          result[0] += 0.003464285315254766;
        } else {
          if (LIKELY(false || (data[0].qvalue <= 74))) {
            result[0] += 0.001344884231078165;
          } else {
            result[0] += 0.007599763760233628;
          }
        }
      }
    }
  } else {
    if (LIKELY(false || (data[2].qvalue <= 148))) {
      if (LIKELY(false || (data[0].qvalue <= 80))) {
        if (LIKELY(false || (data[2].qvalue <= 144))) {
          if (UNLIKELY(false || (data[3].qvalue <= 24))) {
            result[0] += 0.008852681654884861;
          } else {
            result[0] += 0.002711342951735739;
          }
        } else {
          if (LIKELY(false || (data[3].qvalue <= 58))) {
            if (LIKELY(false || (data[0].qvalue <= 30))) {
              result[0] += 0.004675440064623506;
            } else {
              result[0] += 0.007392307943903377;
            }
          } else {
            result[0] += -0.001327311393919067;
          }
        }
      } else {
        result[0] += 0.013657623668096947;
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 44))) {
        if (UNLIKELY(false || (data[2].qvalue <= 150))) {
          result[0] += 0.0063594791059816335;
        } else {
          result[0] += 0.008437232533364402;
        }
      } else {
        if (UNLIKELY(false || (data[2].qvalue <= 152))) {
          if (UNLIKELY(false || (data[0].qvalue <= 38))) {
            result[0] += -0.007575196287639085;
          } else {
            result[0] += 0.010362936929949294;
          }
        } else {
          if (LIKELY(false || (data[3].qvalue <= 58))) {
            if (UNLIKELY(false || (data[0].qvalue <= 38))) {
              result[0] += 0.010091268861320833;
            } else {
              result[0] += 0.017766873342447524;
            }
          } else {
            result[0] += 0.029082956597010896;
          }
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 36))) {
    if (LIKELY(false || (data[1].qvalue <= 20))) {
      if (LIKELY(false || (data[1].qvalue <= 0))) {
        result[0] += -0.000321332335062971;
      } else {
        if (UNLIKELY(false || (data[0].qvalue <= 6))) {
          if (UNLIKELY(false || (data[0].qvalue <= 2))) {
            result[0] += -0.0002294363570388214;
          } else {
            if (LIKELY(false || (data[1].qvalue <= 2))) {
              result[0] += 0.0005885804579918629;
            } else {
              result[0] += 0.0017855147306627796;
            }
          }
        } else {
          result[0] += -0.00022345013791362544;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 16))) {
        if (LIKELY(false || (data[0].qvalue <= 12))) {
          result[0] += 2.4700241334642348e-05;
        } else {
          if (LIKELY(false || (data[0].qvalue <= 42))) {
            result[0] += -0.00020036238717468835;
          } else {
            result[0] += 0.0012242986912053152;
          }
        }
      } else {
        if (UNLIKELY(false || (data[0].qvalue <= 18))) {
          result[0] += 0.000495591293193325;
        } else {
          if (LIKELY(false || (data[1].qvalue <= 58))) {
            result[0] += 8.398870862786576e-05;
          } else {
            if (UNLIKELY(false || (data[1].qvalue <= 62))) {
              result[0] += 0.0009488573573794414;
            } else {
              result[0] += 0.00024307072370918857;
            }
          }
        }
      }
    }
  } else {
    if (UNLIKELY(false || (data[1].qvalue <= 28))) {
      result[0] += 0.0033778298438371006;
    } else {
      if (UNLIKELY(false || (data[1].qvalue <= 32))) {
        result[0] += -0.00018653029821255553;
      } else {
        if (LIKELY(false || (data[3].qvalue <= 48))) {
          result[0] += 0.0013578038474610836;
        } else {
          if (UNLIKELY(false || (data[0].qvalue <= 72))) {
            result[0] += 0.007771589888056477;
          } else {
            if (LIKELY(false || (data[1].qvalue <= 78))) {
              result[0] += 0.002061396852944069;
            } else {
              result[0] += 0.003952529031744175;
            }
          }
        }
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 138))) {
    if (LIKELY(false || (data[2].qvalue <= 94))) {
      if (LIKELY(false || (data[1].qvalue <= 66))) {
        if (LIKELY(false || (data[2].qvalue <= 64))) {
          if (LIKELY(false || (data[1].qvalue <= 22))) {
            if (LIKELY(false || (data[3].qvalue <= 38))) {
              result[0] += -0.0002819994363762382;
            } else {
              result[0] += -0.004239400330393072;
            }
          } else {
            result[0] += -0.00013827393706384455;
          }
        } else {
          if (UNLIKELY(false || (data[3].qvalue <= 14))) {
            if (LIKELY(false || (data[2].qvalue <= 80))) {
              result[0] += 7.513439279655798e-05;
            } else {
              result[0] += 0.0007472308363421268;
            }
          } else {
            if (LIKELY(false || (data[1].qvalue <= 30))) {
              result[0] += -0.00026325700404833315;
            } else {
              result[0] += 0.00020351968979620772;
            }
          }
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 56))) {
          result[0] += 0.000994229445321635;
        } else {
          result[0] += 0.002762669975718151;
        }
      }
    } else {
      if (LIKELY(false || (data[1].qvalue <= 60))) {
        if (LIKELY(false || (data[2].qvalue <= 122))) {
          if (LIKELY(false || (data[1].qvalue <= 48))) {
            if (LIKELY(false || (data[3].qvalue <= 38))) {
              result[0] += 0.00048738447627366677;
            } else {
              result[0] += -0.0008016673054020819;
            }
          } else {
            result[0] += 0.0013491982872427825;
          }
        } else {
          if (UNLIKELY(false || (data[3].qvalue <= 36))) {
            result[0] += 0.003373441637139947;
          } else {
            result[0] += 0.0011426012802899458;
          }
        }
      } else {
        if (LIKELY(false || (data[1].qvalue <= 74))) {
          if (UNLIKELY(false || (data[1].qvalue <= 62))) {
            result[0] += 0.004820049065765553;
          } else {
            if (UNLIKELY(false || (data[1].qvalue <= 70))) {
              result[0] += -0.0005516126023022891;
            } else {
              result[0] += 0.002897319749185366;
            }
          }
        } else {
          result[0] += 0.0058772611298245165;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[2].qvalue <= 148))) {
      if (LIKELY(false || (data[1].qvalue <= 70))) {
        if (LIKELY(false || (data[2].qvalue <= 146))) {
          if (LIKELY(false || (data[3].qvalue <= 54))) {
            if (UNLIKELY(false || (data[3].qvalue <= 24))) {
              result[0] += 0.008367746274452657;
            } else {
              result[0] += 0.0022013121979746215;
            }
          } else {
            result[0] += -0.0021588341517871502;
          }
        } else {
          result[0] += 0.004543503641829177;
        }
      } else {
        if (LIKELY(false || (data[1].qvalue <= 78))) {
          result[0] += 0.007435827781432688;
        } else {
          result[0] += 0.01190804336570028;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 44))) {
        if (UNLIKELY(false || (data[2].qvalue <= 150))) {
          result[0] += 0.005390048627758562;
        } else {
          result[0] += 0.007261544693340969;
        }
      } else {
        if (UNLIKELY(false || (data[2].qvalue <= 152))) {
          if (UNLIKELY(false || (data[1].qvalue <= 72))) {
            result[0] += -0.005337102150777355;
          } else {
            result[0] += 0.008993863393954641;
          }
        } else {
          if (LIKELY(false || (data[3].qvalue <= 58))) {
            if (UNLIKELY(false || (data[1].qvalue <= 64))) {
              result[0] += 0.008333794155773979;
            } else {
              result[0] += 0.01566137192688039;
            }
          } else {
            result[0] += 0.025425005405029257;
          }
        }
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 138))) {
    if (LIKELY(false || (data[2].qvalue <= 108))) {
      if (LIKELY(false || (data[0].qvalue <= 74))) {
        if (LIKELY(false || (data[2].qvalue <= 76))) {
          if (LIKELY(false || (data[2].qvalue <= 24))) {
            if (LIKELY(false || (data[3].qvalue <= 34))) {
              result[0] += -0.0002526916639581458;
            } else {
              result[0] += -0.0008324735013533947;
            }
          } else {
            if (LIKELY(false || (data[0].qvalue <= 16))) {
              result[0] += -3.7354343032152795e-05;
            } else {
              result[0] += -0.00023134344820509463;
            }
          }
        } else {
          if (UNLIKELY(false || (data[3].qvalue <= 12))) {
            result[0] += 0.0006301467008408141;
          } else {
            if (LIKELY(false || (data[3].qvalue <= 30))) {
              result[0] += 0.00017263467395112425;
            } else {
              result[0] += -0.000222732211427673;
            }
          }
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 56))) {
          if (UNLIKELY(false || (data[2].qvalue <= 4))) {
            result[0] += -0.0009052675927216569;
          } else {
            result[0] += 0.0010961202845499573;
          }
        } else {
          result[0] += 0.0034438631349503835;
        }
      }
    } else {
      if (LIKELY(false || (data[0].qvalue <= 78))) {
        if (LIKELY(false || (data[0].qvalue <= 48))) {
          if (UNLIKELY(false || (data[2].qvalue <= 124))) {
            result[0] += 0.0006888856342971111;
          } else {
            result[0] += 0.0011003002879949233;
          }
        } else {
          result[0] += 0.0021751931006627804;
        }
      } else {
        result[0] += 0.006117324596259431;
      }
    }
  } else {
    if (LIKELY(false || (data[2].qvalue <= 148))) {
      if (LIKELY(false || (data[0].qvalue <= 80))) {
        if (LIKELY(false || (data[2].qvalue <= 146))) {
          if (LIKELY(false || (data[3].qvalue <= 54))) {
            if (LIKELY(false || (data[2].qvalue <= 142))) {
              result[0] += 0.001886015702980746;
            } else {
              result[0] += 0.0028879528128823518;
            }
          } else {
            result[0] += -0.0019439231080232771;
          }
        } else {
          result[0] += 0.00409802749099343;
        }
      } else {
        result[0] += 0.010724971462071377;
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 44))) {
        if (UNLIKELY(false || (data[2].qvalue <= 150))) {
          result[0] += 0.004851273450709325;
        } else {
          if (LIKELY(false || (data[2].qvalue <= 152))) {
            result[0] += 0.005993594323114087;
          } else {
            result[0] += 0.007126830483296861;
          }
        }
      } else {
        if (UNLIKELY(false || (data[2].qvalue <= 152))) {
          if (UNLIKELY(false || (data[0].qvalue <= 38))) {
            result[0] += -0.006591833172632115;
          } else {
            result[0] += 0.007745557909157859;
          }
        } else {
          if (LIKELY(false || (data[3].qvalue <= 58))) {
            if (UNLIKELY(false || (data[0].qvalue <= 38))) {
              result[0] += 0.007513856667529552;
            } else {
              result[0] += 0.014099994844270129;
            }
          } else {
            result[0] += 0.022901478010549474;
          }
        }
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 138))) {
    if (LIKELY(false || (data[2].qvalue <= 100))) {
      if (LIKELY(false || (data[1].qvalue <= 66))) {
        if (LIKELY(false || (data[2].qvalue <= 64))) {
          if (LIKELY(false || (data[1].qvalue <= 22))) {
            result[0] += -0.00023332139181383586;
          } else {
            if (LIKELY(false || (data[2].qvalue <= 50))) {
              result[0] += -0.00017980571342344313;
            } else {
              result[0] += -2.0718875487255795e-05;
            }
          }
        } else {
          if (LIKELY(false || (data[1].qvalue <= 48))) {
            if (UNLIKELY(false || (data[1].qvalue <= 8))) {
              result[0] += -0.0002518917005409201;
            } else {
              result[0] += 7.227477975380027e-05;
            }
          } else {
            if (LIKELY(false || (data[1].qvalue <= 60))) {
              result[0] += 0.0005108640832635015;
            } else {
              result[0] += 0.0016008570594736012;
            }
          }
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 70))) {
          if (UNLIKELY(false || (data[2].qvalue <= 2))) {
            result[0] += -0.0004895282097672478;
          } else {
            result[0] += 0.0009271045263167995;
          }
        } else {
          result[0] += 0.003029910672478344;
        }
      }
    } else {
      if (LIKELY(false || (data[1].qvalue <= 48))) {
        if (UNLIKELY(false || (data[2].qvalue <= 122))) {
          result[0] += 0.00040218816684723627;
        } else {
          if (UNLIKELY(false || (data[1].qvalue <= 12))) {
            result[0] += 0.0032437816301035304;
          } else {
            result[0] += 0.0009341826233583182;
          }
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 116))) {
          result[0] += 0.0014590413273968005;
        } else {
          result[0] += 0.0029815343815250677;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[2].qvalue <= 148))) {
      if (LIKELY(false || (data[1].qvalue <= 70))) {
        if (LIKELY(false || (data[2].qvalue <= 146))) {
          if (LIKELY(false || (data[0].qvalue <= 32))) {
            if (UNLIKELY(false || (data[1].qvalue <= 14))) {
              result[0] += 0.006437739111759044;
            } else {
              result[0] += 0.0017784105014231725;
            }
          } else {
            if (LIKELY(false || (data[0].qvalue <= 36))) {
              result[0] += -0.001771490979701454;
            } else {
              result[0] += 0.0020923301462197823;
            }
          }
        } else {
          result[0] += 0.0036797237719870812;
        }
      } else {
        result[0] += 0.007668662148627889;
      }
    } else {
      if (LIKELY(false || (data[0].qvalue <= 24))) {
        if (UNLIKELY(false || (data[2].qvalue <= 150))) {
          result[0] += 0.004320617911165859;
        } else {
          result[0] += 0.005898161657802002;
        }
      } else {
        if (UNLIKELY(false || (data[2].qvalue <= 152))) {
          if (UNLIKELY(false || (data[2].qvalue <= 150))) {
            result[0] += 0;
          } else {
            result[0] += 0.0069151463115944434;
          }
        } else {
          if (UNLIKELY(false || (data[0].qvalue <= 36))) {
            result[0] += 0.020628421008308875;
          } else {
            result[0] += 0.012233663540640245;
          }
        }
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 138))) {
    if (LIKELY(false || (data[2].qvalue <= 96))) {
      if (LIKELY(false || (data[1].qvalue <= 66))) {
        if (LIKELY(false || (data[2].qvalue <= 50))) {
          if (LIKELY(false || (data[3].qvalue <= 34))) {
            result[0] += -0.0001966547586316513;
          } else {
            if (UNLIKELY(false || (data[3].qvalue <= 52))) {
              result[0] += -0.001949253703254182;
            } else {
              result[0] += -0.0003961865052170259;
            }
          }
        } else {
          if (LIKELY(false || (data[1].qvalue <= 60))) {
            if (LIKELY(false || (data[2].qvalue <= 80))) {
              result[0] += -8.702945155336081e-05;
            } else {
              result[0] += 0.00010101002842960764;
            }
          } else {
            result[0] += 0.0011539892201556142;
          }
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 46))) {
          result[0] += 0.0006699104798973902;
        } else {
          result[0] += 0.0016097103650539448;
        }
      }
    } else {
      if (LIKELY(false || (data[1].qvalue <= 48))) {
        if (LIKELY(false || (data[2].qvalue <= 122))) {
          if (LIKELY(false || (data[3].qvalue <= 38))) {
            result[0] += 0.00037384006745287045;
          } else {
            result[0] += -0.00078417496305398;
          }
        } else {
          if (UNLIKELY(false || (data[3].qvalue <= 36))) {
            result[0] += 0.002583360550126892;
          } else {
            result[0] += 0.0008286684098961815;
          }
        }
      } else {
        if (LIKELY(false || (data[1].qvalue <= 74))) {
          if (LIKELY(false || (data[2].qvalue <= 118))) {
            result[0] += 0.001147831022132617;
          } else {
            if (UNLIKELY(false || (data[3].qvalue <= 20))) {
              result[0] += 0.006146182855694659;
            } else {
              result[0] += 0.002161465650341462;
            }
          }
        } else {
          result[0] += 0.004769212968618104;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[2].qvalue <= 148))) {
      if (LIKELY(false || (data[1].qvalue <= 70))) {
        if (LIKELY(false || (data[2].qvalue <= 146))) {
          if (LIKELY(false || (data[3].qvalue <= 54))) {
            if (UNLIKELY(false || (data[3].qvalue <= 28))) {
              result[0] += 0.006289090271765258;
            } else {
              result[0] += 0.0015995773242235114;
            }
          } else {
            result[0] += -0.0015852486767579577;
          }
        } else {
          result[0] += 0.0033119101068960115;
        }
      } else {
        result[0] += 0.006903666361705358;
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 44))) {
        if (LIKELY(false || (data[2].qvalue <= 152))) {
          result[0] += 0.004350684674853294;
        } else {
          result[0] += 0.0058160004156852805;
        }
      } else {
        if (UNLIKELY(false || (data[2].qvalue <= 152))) {
          if (UNLIKELY(false || (data[1].qvalue <= 72))) {
            result[0] += -0.005029844201946011;
          } else {
            result[0] += 0.006639408804302992;
          }
        } else {
          if (LIKELY(false || (data[3].qvalue <= 58))) {
            if (UNLIKELY(false || (data[1].qvalue <= 64))) {
              result[0] += 0.00619428725665315;
            } else {
              result[0] += 0.011474633151106254;
            }
          } else {
            result[0] += 0.01858097359565879;
          }
        }
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 138))) {
    if (LIKELY(false || (data[2].qvalue <= 108))) {
      if (LIKELY(false || (data[0].qvalue <= 46))) {
        if (LIKELY(false || (data[2].qvalue <= 76))) {
          if (LIKELY(false || (data[2].qvalue <= 20))) {
            if (LIKELY(false || (data[3].qvalue <= 38))) {
              result[0] += -0.00019940358387546183;
            } else {
              result[0] += -0.000703113116348102;
            }
          } else {
            if (LIKELY(false || (data[0].qvalue <= 16))) {
              result[0] += -4.638458464186633e-05;
            } else {
              result[0] += -0.0002294784137776043;
            }
          }
        } else {
          if (UNLIKELY(false || (data[3].qvalue <= 12))) {
            result[0] += 0.0005471125373971178;
          } else {
            if (LIKELY(false || (data[3].qvalue <= 30))) {
              result[0] += 9.670414976687889e-05;
            } else {
              result[0] += -0.00021767801684133135;
            }
          }
        }
      } else {
        if (UNLIKELY(false || (data[2].qvalue <= 8))) {
          result[0] += -3.166344582237319e-05;
        } else {
          if (LIKELY(false || (data[0].qvalue <= 78))) {
            if (UNLIKELY(false || (data[0].qvalue <= 58))) {
              result[0] += 0.0008152326933001966;
            } else {
              result[0] += 0.00025174464498736135;
            }
          } else {
            if (LIKELY(false || (data[0].qvalue <= 80))) {
              result[0] += 0.0013971405413898312;
            } else {
              result[0] += -0.0006916749685131233;
            }
          }
        }
      }
    } else {
      if (LIKELY(false || (data[0].qvalue <= 78))) {
        if (LIKELY(false || (data[0].qvalue <= 42))) {
          result[0] += 0.0006645427754437797;
        } else {
          result[0] += 0.0013310551905353592;
        }
      } else {
        result[0] += 0.004798737931736182;
      }
    }
  } else {
    if (LIKELY(false || (data[2].qvalue <= 150))) {
      if (LIKELY(false || (data[2].qvalue <= 146))) {
        if (LIKELY(false || (data[0].qvalue <= 80))) {
          if (LIKELY(false || (data[3].qvalue <= 54))) {
            if (LIKELY(false || (data[2].qvalue <= 142))) {
              result[0] += 0.0013586733582895156;
            } else {
              result[0] += 0.002186283295362004;
            }
          } else {
            result[0] += -0.0014274378502657555;
          }
        } else {
          if (LIKELY(false || (data[2].qvalue <= 140))) {
            result[0] += 0.006650116111876395;
          } else {
            result[0] += 0.013644770354148933;
          }
        }
      } else {
        if (LIKELY(false || (data[3].qvalue <= 44))) {
          result[0] += 0.0032603013506928773;
        } else {
          result[0] += -0.010240866610706277;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 58))) {
        if (LIKELY(false || (data[0].qvalue <= 68))) {
          if (LIKELY(false || (data[3].qvalue <= 48))) {
            result[0] += 0.004779958775547548;
          } else {
            result[0] += 0.00686445672987769;
          }
        } else {
          result[0] += 0.010596439063715636;
        }
      } else {
        result[0] += 0.016736742650200403;
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 138))) {
    if (LIKELY(false || (data[2].qvalue <= 104))) {
      if (LIKELY(false || (data[1].qvalue <= 66))) {
        if (LIKELY(false || (data[2].qvalue <= 74))) {
          if (LIKELY(false || (data[1].qvalue <= 22))) {
            result[0] += -0.00017265302736832624;
          } else {
            result[0] += -6.590425058702988e-05;
          }
        } else {
          if (UNLIKELY(false || (data[1].qvalue <= 8))) {
            result[0] += -0.00021500489047340408;
          } else {
            if (UNLIKELY(false || (data[0].qvalue <= 28))) {
              result[0] += 0.0003260276251636524;
            } else {
              result[0] += 1.1618765633385006e-05;
            }
          }
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 56))) {
          result[0] += 0.0006091551148605127;
        } else {
          result[0] += 0.002041106193822838;
        }
      }
    } else {
      if (LIKELY(false || (data[1].qvalue <= 48))) {
        if (UNLIKELY(false || (data[2].qvalue <= 124))) {
          result[0] += 0.0003264125370269558;
        } else {
          result[0] += 0.0007305621801641551;
        }
      } else {
        if (LIKELY(false || (data[1].qvalue <= 74))) {
          if (LIKELY(false || (data[0].qvalue <= 64))) {
            result[0] += 0.0015308893471321235;
          } else {
            result[0] += -0.0011053771592844164;
          }
        } else {
          result[0] += 0.004019341238603161;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[2].qvalue <= 150))) {
      if (LIKELY(false || (data[2].qvalue <= 146))) {
        if (LIKELY(false || (data[1].qvalue <= 70))) {
          result[0] += 0.0012825109063898839;
        } else {
          result[0] += 0.005737490966667883;
        }
      } else {
        result[0] += 0.002875416523570694;
      }
    } else {
      if (LIKELY(false || (data[0].qvalue <= 24))) {
        result[0] += 0.0043220277108669625;
      } else {
        if (UNLIKELY(false || (data[0].qvalue <= 36))) {
          result[0] += 0.01467062094307446;
        } else {
          if (UNLIKELY(false || (data[2].qvalue <= 152))) {
            result[0] += 0.00487359498506815;
          } else {
            result[0] += 0.009006784051361535;
          }
        }
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 138))) {
    if (LIKELY(false || (data[2].qvalue <= 108))) {
      if (LIKELY(false || (data[1].qvalue <= 60))) {
        if (LIKELY(false || (data[2].qvalue <= 82))) {
          if (LIKELY(false || (data[2].qvalue <= 12))) {
            result[0] += -0.0001702686007892128;
          } else {
            if (LIKELY(false || (data[0].qvalue <= 16))) {
              result[0] += -4.754655862563027e-05;
            } else {
              result[0] += -0.00017474323658447127;
            }
          }
        } else {
          if (UNLIKELY(false || (data[1].qvalue <= 18))) {
            result[0] += -0.00018950144326743167;
          } else {
            if (UNLIKELY(false || (data[1].qvalue <= 24))) {
              result[0] += 0.0005661111931274635;
            } else {
              result[0] += 0.00014359699750018804;
            }
          }
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 44))) {
          if (UNLIKELY(false || (data[2].qvalue <= 4))) {
            result[0] += -9.242813453442713e-05;
          } else {
            result[0] += 0.0004318537730927437;
          }
        } else {
          result[0] += 0.000996025787596676;
        }
      }
    } else {
      if (LIKELY(false || (data[1].qvalue <= 42))) {
        if (UNLIKELY(false || (data[1].qvalue <= 10))) {
          result[0] += 0.0013570211830916067;
        } else {
          if (UNLIKELY(false || (data[1].qvalue <= 34))) {
            result[0] += 6.900395934849614e-05;
          } else {
            result[0] += 0.000526718399212838;
          }
        }
      } else {
        if (LIKELY(false || (data[1].qvalue <= 74))) {
          result[0] += 0.0012271596277449244;
        } else {
          result[0] += 0.0038837618064354986;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[2].qvalue <= 148))) {
      if (LIKELY(false || (data[1].qvalue <= 70))) {
        if (LIKELY(false || (data[2].qvalue <= 146))) {
          result[0] += 0.0011542642646145598;
        } else {
          result[0] += 0.0023728018844858226;
        }
      } else {
        result[0] += 0.00520880423871823;
      }
    } else {
      if (LIKELY(false || (data[0].qvalue <= 24))) {
        if (LIKELY(false || (data[2].qvalue <= 152))) {
          result[0] += 0.003135293929630502;
        } else {
          result[0] += 0.004343381199101189;
        }
      } else {
        if (UNLIKELY(false || (data[2].qvalue <= 152))) {
          if (UNLIKELY(false || (data[0].qvalue <= 38))) {
            result[0] += 0.0005685181748121977;
          } else {
            result[0] += 0.004925780552866418;
          }
        } else {
          if (UNLIKELY(false || (data[0].qvalue <= 36))) {
            result[0] += 0.013619444400285816;
          } else {
            result[0] += 0.008108635586280942;
          }
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 36))) {
    if (LIKELY(false || (data[1].qvalue <= 20))) {
      result[0] += -0.00012650171567985005;
    } else {
      if (LIKELY(false || (data[0].qvalue <= 48))) {
        result[0] += -4.938551105130385e-07;
      } else {
        result[0] += 0.00023777414317281933;
      }
    }
  } else {
    if (UNLIKELY(false || (data[1].qvalue <= 28))) {
      result[0] += 0.0014209088284525466;
    } else {
      if (UNLIKELY(false || (data[1].qvalue <= 32))) {
        result[0] += -0.00013475701157859363;
      } else {
        if (LIKELY(false || (data[3].qvalue <= 48))) {
          result[0] += 0.0005692054876058483;
        } else {
          if (UNLIKELY(false || (data[0].qvalue <= 72))) {
            result[0] += 0.002924335912848483;
          } else {
            result[0] += 0.0011432533753282545;
          }
        }
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 142))) {
    if (LIKELY(false || (data[2].qvalue <= 108))) {
      if (LIKELY(false || (data[2].qvalue <= 26))) {
        result[0] += -0.00012932196425586976;
      } else {
        if (LIKELY(false || (data[0].qvalue <= 48))) {
          if (LIKELY(false || (data[2].qvalue <= 80))) {
            result[0] += -5.748038407974522e-05;
          } else {
            result[0] += 9.26994872808481e-05;
          }
        } else {
          if (UNLIKELY(false || (data[0].qvalue <= 58))) {
            result[0] += 0.0008936605731170518;
          } else {
            if (LIKELY(false || (data[0].qvalue <= 74))) {
              result[0] += -1.2440258612188177e-05;
            } else {
              result[0] += 0.0005585540355690144;
            }
          }
        }
      }
    } else {
      if (LIKELY(false || (data[0].qvalue <= 78))) {
        if (UNLIKELY(false || (data[2].qvalue <= 132))) {
          result[0] += 0.00040803348778467856;
        } else {
          if (UNLIKELY(false || (data[3].qvalue <= 36))) {
            result[0] += 0.0025997388488009357;
          } else {
            if (LIKELY(false || (data[3].qvalue <= 54))) {
              result[0] += 0.0007607395473800554;
            } else {
              result[0] += -0.006367553819648243;
            }
          }
        }
      } else {
        result[0] += 0.004347370506359142;
      }
    }
  } else {
    if (LIKELY(false || (data[2].qvalue <= 152))) {
      if (LIKELY(false || (data[3].qvalue <= 58))) {
        if (LIKELY(false || (data[2].qvalue <= 148))) {
          if (LIKELY(false || (data[0].qvalue <= 80))) {
            if (UNLIKELY(false || (data[3].qvalue <= 26))) {
              result[0] += 0.008218266539471714;
            } else {
              result[0] += 0.001689778047040225;
            }
          } else {
            result[0] += 0.010197310490906239;
          }
        } else {
          result[0] += 0.002771954438342974;
        }
      } else {
        result[0] += -0.00263078958160482;
      }
    } else {
      if (LIKELY(false || (data[0].qvalue <= 32))) {
        result[0] += 0.003762501649809371;
      } else {
        if (UNLIKELY(false || (data[0].qvalue <= 36))) {
          result[0] += 0.011977412732067837;
        } else {
          result[0] += 0.007180131322773255;
        }
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 142))) {
    if (LIKELY(false || (data[2].qvalue <= 110))) {
      if (LIKELY(false || (data[1].qvalue <= 60))) {
        if (LIKELY(false || (data[2].qvalue <= 84))) {
          if (LIKELY(false || (data[3].qvalue <= 38))) {
            result[0] += -9.548885583609136e-05;
          } else {
            if (UNLIKELY(false || (data[3].qvalue <= 40))) {
              result[0] += -0.003540081420843614;
            } else {
              result[0] += -0.00024649854685375276;
            }
          }
        } else {
          if (UNLIKELY(false || (data[3].qvalue <= 20))) {
            result[0] += 0.0003053226725768677;
          } else {
            result[0] += -6.602264761802435e-05;
          }
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 68))) {
          if (UNLIKELY(false || (data[2].qvalue <= 4))) {
            result[0] += -9.762664695776116e-05;
          } else {
            result[0] += 0.00038670152054575205;
          }
        } else {
          result[0] += 0.0014758023283282069;
        }
      }
    } else {
      if (LIKELY(false || (data[1].qvalue <= 76))) {
        if (LIKELY(false || (data[2].qvalue <= 138))) {
          if (LIKELY(false || (data[1].qvalue <= 42))) {
            if (LIKELY(false || (data[3].qvalue <= 38))) {
              result[0] += 0.0003950345531674714;
            } else {
              result[0] += -0.0006780454492008814;
            }
          } else {
            result[0] += 0.0011275851879113837;
          }
        } else {
          result[0] += 0.000772236033582265;
        }
      } else {
        result[0] += 0.004143313796252665;
      }
    }
  } else {
    if (LIKELY(false || (data[2].qvalue <= 150))) {
      if (LIKELY(false || (data[3].qvalue <= 58))) {
        if (UNLIKELY(false || (data[3].qvalue <= 26))) {
          result[0] += 0.009037786461412907;
        } else {
          if (LIKELY(false || (data[1].qvalue <= 78))) {
            result[0] += 0.0016797847044042091;
          } else {
            result[0] += 0.009218368388153613;
          }
        }
      } else {
        result[0] += -0.002368890210880294;
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 58))) {
        if (LIKELY(false || (data[1].qvalue <= 64))) {
          result[0] += 0.0030468957829948803;
        } else {
          if (UNLIKELY(false || (data[2].qvalue <= 152))) {
            result[0] += 0.003570656648731198;
          } else {
            result[0] += 0.006838003122049602;
          }
        }
      } else {
        result[0] += 0.010788609192028134;
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 144))) {
    if (LIKELY(false || (data[2].qvalue <= 116))) {
      if (LIKELY(false || (data[1].qvalue <= 48))) {
        if (LIKELY(false || (data[2].qvalue <= 50))) {
          result[0] += -0.00010835034231308543;
        } else {
          result[0] += -3.592209291896241e-06;
        }
      } else {
        if (UNLIKELY(false || (data[2].qvalue <= 10))) {
          result[0] += -6.758697818328856e-05;
        } else {
          if (LIKELY(false || (data[2].qvalue <= 68))) {
            result[0] += 0.0002567234765659431;
          } else {
            result[0] += 0.0006684446514199459;
          }
        }
      }
    } else {
      if (LIKELY(false || (data[1].qvalue <= 38))) {
        if (UNLIKELY(false || (data[1].qvalue <= 4))) {
          result[0] += 0.0019020060332526339;
        } else {
          if (UNLIKELY(false || (data[2].qvalue <= 126))) {
            result[0] += 0.00014095668063188527;
          } else {
            result[0] += 0.0006037159852907501;
          }
        }
      } else {
        if (LIKELY(false || (data[0].qvalue <= 74))) {
          result[0] += 0.001560769119473446;
        } else {
          result[0] += 0.004308113069032753;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[2].qvalue <= 152))) {
      if (LIKELY(false || (data[2].qvalue <= 150))) {
        result[0] += 0.001609181337370336;
      } else {
        result[0] += 0.0025188474001257414;
      }
    } else {
      if (LIKELY(false || (data[0].qvalue <= 32))) {
        result[0] += 0.0030819144903457287;
      } else {
        if (UNLIKELY(false || (data[0].qvalue <= 36))) {
          result[0] += 0.009717799607497545;
        } else {
          result[0] += 0.005811008536776794;
        }
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 142))) {
    if (LIKELY(false || (data[2].qvalue <= 108))) {
      if (LIKELY(false || (data[2].qvalue <= 26))) {
        result[0] += -9.633149362070105e-05;
      } else {
        if (LIKELY(false || (data[3].qvalue <= 46))) {
          if (LIKELY(false || (data[3].qvalue <= 42))) {
            result[0] += -2.3786842856587217e-06;
          } else {
            result[0] += -0.0008784412376800542;
          }
        } else {
          if (LIKELY(false || (data[3].qvalue <= 52))) {
            result[0] += 0.0005592856904334903;
          } else {
            result[0] += -0.0010997740778614094;
          }
        }
      }
    } else {
      if (LIKELY(false || (data[0].qvalue <= 78))) {
        if (LIKELY(false || (data[2].qvalue <= 134))) {
          result[0] += 0.0003003870717462951;
        } else {
          if (UNLIKELY(false || (data[3].qvalue <= 36))) {
            result[0] += 0.0021186501027940312;
          } else {
            result[0] += 0.0005632598779634045;
          }
        }
      } else {
        result[0] += 0.003155110609239992;
      }
    }
  } else {
    if (LIKELY(false || (data[2].qvalue <= 152))) {
      if (LIKELY(false || (data[3].qvalue <= 58))) {
        if (UNLIKELY(false || (data[3].qvalue <= 26))) {
          if (LIKELY(false || (data[3].qvalue <= 22))) {
            result[0] += 0.002531461699501328;
          } else {
            result[0] += 0.013085974014940716;
          }
        } else {
          if (LIKELY(false || (data[2].qvalue <= 148))) {
            result[0] += 0.0012521484883360095;
          } else {
            result[0] += 0.002029939821163165;
          }
        }
      } else {
        result[0] += -0.002291005669175251;
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 44))) {
        result[0] += 0.0027415007075521983;
      } else {
        if (LIKELY(false || (data[3].qvalue <= 58))) {
          result[0] += 0.00517479080318933;
        } else {
          result[0] += 0.008753272564108693;
        }
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 36))) {
    if (LIKELY(false || (data[1].qvalue <= 20))) {
      result[0] += -7.805033787009575e-05;
    } else {
      result[0] += 2.3801570604239546e-05;
    }
  } else {
    if (UNLIKELY(false || (data[1].qvalue <= 28))) {
      result[0] += 0.0008308640887867767;
    } else {
      if (LIKELY(false || (data[1].qvalue <= 36))) {
        if (UNLIKELY(false || (data[1].qvalue <= 32))) {
          result[0] += -0.0001255589210486739;
        } else {
          result[0] += 0.0002974589212792894;
        }
      } else {
        if (UNLIKELY(false || (data[0].qvalue <= 66))) {
          result[0] += 0.001434875231823905;
        } else {
          if (LIKELY(false || (data[0].qvalue <= 78))) {
            result[0] += 0.00015393892861622098;
          } else {
            result[0] += 0.0008355777768832701;
          }
        }
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 144))) {
    if (LIKELY(false || (data[2].qvalue <= 100))) {
      if (LIKELY(false || (data[0].qvalue <= 46))) {
        if (LIKELY(false || (data[2].qvalue <= 80))) {
          result[0] += -7.09997632022635e-05;
        } else {
          if (LIKELY(false || (data[2].qvalue <= 92))) {
            if (LIKELY(false || (data[2].qvalue <= 90))) {
              result[0] += 6.700198529498124e-05;
            } else {
              result[0] += 0.0006739111265931478;
            }
          } else {
            result[0] += -7.873394714685852e-05;
          }
        }
      } else {
        if (UNLIKELY(false || (data[2].qvalue <= 0))) {
          result[0] += -0.00043781627567509543;
        } else {
          result[0] += 0.00020885997574915155;
        }
      }
    } else {
      if (LIKELY(false || (data[0].qvalue <= 80))) {
        if (UNLIKELY(false || (data[2].qvalue <= 130))) {
          result[0] += 0.00021999474610715451;
        } else {
          if (UNLIKELY(false || (data[3].qvalue <= 36))) {
            result[0] += 0.001993992760088811;
          } else {
            result[0] += 0.00044991030401214167;
          }
        }
      } else {
        result[0] += 0.004000181025335582;
      }
    }
  } else {
    if (LIKELY(false || (data[2].qvalue <= 152))) {
      if (LIKELY(false || (data[3].qvalue <= 58))) {
        result[0] += 0.0015042104367088394;
      } else {
        result[0] += -0.0022578272852008534;
      }
    } else {
      if (LIKELY(false || (data[0].qvalue <= 32))) {
        result[0] += 0.002410829276957461;
      } else {
        result[0] += 0.0055512269901312646;
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 146))) {
    if (LIKELY(false || (data[2].qvalue <= 104))) {
      if (UNLIKELY(false || (data[2].qvalue <= 12))) {
        result[0] += -8.496817920340536e-05;
      } else {
        if (LIKELY(false || (data[0].qvalue <= 48))) {
          result[0] += -1.7540259968379833e-05;
        } else {
          result[0] += 0.00025745579920166244;
        }
      }
    } else {
      if (LIKELY(false || (data[0].qvalue <= 78))) {
        if (LIKELY(false || (data[2].qvalue <= 138))) {
          result[0] += 0.00023710498818372454;
        } else {
          result[0] += 0.0005008600118271386;
        }
      } else {
        if (LIKELY(false || (data[2].qvalue <= 140))) {
          result[0] += 0.0021586178965176087;
        } else {
          result[0] += 0.008461107069451827;
        }
      }
    }
  } else {
    if (LIKELY(false || (data[0].qvalue <= 24))) {
      if (LIKELY(false || (data[2].qvalue <= 150))) {
        result[0] += 0.0012443975867339444;
      } else {
        result[0] += 0.0019477678730867058;
      }
    } else {
      if (UNLIKELY(false || (data[2].qvalue <= 152))) {
        result[0] += 0.0019342118146449738;
      } else {
        result[0] += 0.004997239694835386;
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 32))) {
    if (LIKELY(false || (data[1].qvalue <= 16))) {
      result[0] += -6.499770451690952e-05;
    } else {
      result[0] += 9.93532909532843e-06;
    }
  } else {
    if (UNLIKELY(false || (data[1].qvalue <= 28))) {
      result[0] += 0.0005901691687912098;
    } else {
      if (LIKELY(false || (data[1].qvalue <= 70))) {
        result[0] += 0.00015807135019769365;
      } else {
        result[0] += 0.0005244958255941764;
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 146))) {
    if (LIKELY(false || (data[2].qvalue <= 86))) {
      result[0] += -4.676198214734394e-05;
    } else {
      if (LIKELY(false || (data[1].qvalue <= 60))) {
        if (UNLIKELY(false || (data[1].qvalue <= 14))) {
          if (LIKELY(false || (data[2].qvalue <= 118))) {
            result[0] += -0.00010600660656932565;
          } else {
            result[0] += 0.0015161007267231429;
          }
        } else {
          result[0] += 0.0002612619805398477;
        }
      } else {
        result[0] += 0.0015693112213150122;
      }
    }
  } else {
    if (LIKELY(false || (data[3].qvalue <= 44))) {
      result[0] += 0.0013664188809999066;
    } else {
      if (UNLIKELY(false || (data[2].qvalue <= 150))) {
        result[0] += -0.010022143385034394;
      } else {
        result[0] += 0.0037975534515937988;
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 142))) {
    if (LIKELY(false || (data[2].qvalue <= 86))) {
      result[0] += -4.208578851825266e-05;
    } else {
      if (UNLIKELY(false || (data[1].qvalue <= 14))) {
        if (LIKELY(false || (data[2].qvalue <= 116))) {
          result[0] += -0.00011403525794098995;
        } else {
          result[0] += 0.0009114640165412386;
        }
      } else {
        if (LIKELY(false || (data[1].qvalue <= 60))) {
          result[0] += 0.00022356508848759622;
        } else {
          if (UNLIKELY(false || (data[1].qvalue <= 62))) {
            result[0] += 0.0027995720407202294;
          } else {
            if (UNLIKELY(false || (data[1].qvalue <= 70))) {
              result[0] += -0.0008798782950704978;
            } else {
              result[0] += 0.0014448416755765952;
            }
          }
        }
      }
    }
  } else {
    if (LIKELY(false || (data[2].qvalue <= 152))) {
      if (LIKELY(false || (data[1].qvalue <= 70))) {
        result[0] += 0.0008384216911964453;
      } else {
        result[0] += 0.002447881422746419;
      }
    } else {
      if (LIKELY(false || (data[0].qvalue <= 32))) {
        result[0] += 0.0017733230820123455;
      } else {
        result[0] += 0.004110382169902663;
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 146))) {
    if (LIKELY(false || (data[2].qvalue <= 86))) {
      result[0] += -3.787721484857798e-05;
    } else {
      if (LIKELY(false || (data[1].qvalue <= 70))) {
        if (UNLIKELY(false || (data[1].qvalue <= 14))) {
          if (LIKELY(false || (data[2].qvalue <= 118))) {
            result[0] += -8.689697470917008e-05;
          } else {
            result[0] += 0.0012740214207030438;
          }
        } else {
          if (LIKELY(false || (data[3].qvalue <= 54))) {
            result[0] += 0.00021910612133028753;
          } else {
            result[0] += -0.0019394310314736379;
          }
        }
      } else {
        result[0] += 0.001605288614427337;
      }
    }
  } else {
    if (LIKELY(false || (data[2].qvalue <= 152))) {
      if (LIKELY(false || (data[3].qvalue <= 52))) {
        result[0] += 0.0010209367212090006;
      } else {
        result[0] += -0.0070652788154780865;
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 58))) {
        result[0] += 0.001794208969053978;
      } else {
        result[0] += 0.00589450452114028;
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 142))) {
    if (LIKELY(false || (data[2].qvalue <= 86))) {
      result[0] += -3.4089497281904435e-05;
    } else {
      if (LIKELY(false || (data[1].qvalue <= 48))) {
        result[0] += 0.00011859360145590661;
      } else {
        result[0] += 0.0005882149306744061;
      }
    }
  } else {
    if (LIKELY(false || (data[2].qvalue <= 150))) {
      result[0] += 0.0006289461581674275;
    } else {
      if (LIKELY(false || (data[0].qvalue <= 24))) {
        result[0] += 0.0012857444628638962;
      } else {
        if (UNLIKELY(false || (data[0].qvalue <= 36))) {
          result[0] += 0.0053522964003919205;
        } else {
          result[0] += 0.002180858513372534;
        }
      }
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 142))) {
    if (LIKELY(false || (data[2].qvalue <= 122))) {
      if (LIKELY(false || (data[1].qvalue <= 48))) {
        result[0] += -3.231127347736361e-05;
      } else {
        result[0] += 0.0001305453669476947;
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 54))) {
        if (UNLIKELY(false || (data[3].qvalue <= 36))) {
          result[0] += 0.0015725449376088275;
        } else {
          result[0] += 0.0002082949912808661;
        }
      } else {
        result[0] += -0.006547446409612894;
      }
    }
  } else {
    if (LIKELY(false || (data[2].qvalue <= 150))) {
      if (UNLIKELY(false || (data[3].qvalue <= 26))) {
        result[0] += 0.0065892055427092455;
      } else {
        if (LIKELY(false || (data[3].qvalue <= 58))) {
          result[0] += 0.000599515916925125;
        } else {
          result[0] += -0.001937681221394829;
        }
      }
    } else {
      if (LIKELY(false || (data[3].qvalue <= 58))) {
        result[0] += 0.00127534830686723;
      } else {
        result[0] += 0.004778217595251305;
      }
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 38))) {
    result[0] += -1.2646257755117516e-05;
  } else {
    result[0] += 0.0002454812344733204;
  }
  if (LIKELY(false || (data[3].qvalue <= 32))) {
    result[0] += -1.5786346014169378e-05;
  } else {
    result[0] += 0.00016700826916388908;
  }
  if (LIKELY(false || (data[2].qvalue <= 146))) {
    if (LIKELY(false || (data[2].qvalue <= 80))) {
      result[0] += -2.887770604298617e-05;
    } else {
      if (LIKELY(false || (data[0].qvalue <= 78))) {
        if (LIKELY(false || (data[0].qvalue <= 64))) {
          result[0] += 0.00010275584396454631;
        } else {
          result[0] += -0.001987768939441655;
        }
      } else {
        result[0] += 0.0019405763097977533;
      }
    }
  } else {
    if (LIKELY(false || (data[0].qvalue <= 60))) {
      result[0] += 0.0008150039520986579;
    } else {
      result[0] += 0.003428131746816993;
    }
  }
  if (LIKELY(false || (data[3].qvalue <= 38))) {
    result[0] += -9.428440320689368e-06;
  } else {
    result[0] += 0.00018303284915371275;
  }
  if (LIKELY(false || (data[2].qvalue <= 148))) {
    if (LIKELY(false || (data[2].qvalue <= 126))) {
      result[0] += -1.6525789323637374e-05;
    } else {
      result[0] += 0.00019819706642161615;
    }
  } else {
    result[0] += 0.0009033266907285568;
  }
  if (LIKELY(false || (data[2].qvalue <= 142))) {
    if (LIKELY(false || (data[2].qvalue <= 80))) {
      result[0] += -2.3731074344715385e-05;
    } else {
      if (LIKELY(false || (data[3].qvalue <= 54))) {
        if (LIKELY(false || (data[0].qvalue <= 48))) {
          if (UNLIKELY(false || (data[3].qvalue <= 12))) {
            result[0] += 0.0003259490041081464;
          } else {
            result[0] += 3.657792922172123e-05;
          }
        } else {
          if (UNLIKELY(false || (data[0].qvalue <= 50))) {
            result[0] += 0.00406008107199644;
          } else {
            result[0] += 0.0005606708031635196;
          }
        }
      } else {
        result[0] += -0.005101178498007357;
      }
    }
  } else {
    if (LIKELY(false || (data[0].qvalue <= 60))) {
      result[0] += 0.0005373762500503928;
    } else {
      result[0] += 0.0027015011546189507;
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 150))) {
    if (LIKELY(false || (data[2].qvalue <= 128))) {
      result[0] += -1.3661286266953386e-05;
    } else {
      result[0] += 0.00018261991840147;
    }
  } else {
    result[0] += 0.000931821730518591;
  }
  if (LIKELY(false || (data[2].qvalue <= 150))) {
    if (LIKELY(false || (data[2].qvalue <= 126))) {
      result[0] += -1.2529314401332627e-05;
    } else {
      result[0] += 0.00016018169881967258;
    }
  } else {
    result[0] += 0.0008386591836191829;
  }
  if (LIKELY(false || (data[2].qvalue <= 144))) {
    if (LIKELY(false || (data[0].qvalue <= 78))) {
      if (LIKELY(false || (data[2].qvalue <= 64))) {
        result[0] += -2.4302315497459032e-05;
      } else {
        result[0] += 5.194281681207711e-05;
      }
    } else {
      if (LIKELY(false || (data[2].qvalue <= 140))) {
        result[0] += 0.00045022060778704474;
      } else {
        result[0] += 0.006445361257938202;
      }
    }
  } else {
    result[0] += 0.0004861295980061241;
  }
  if (LIKELY(false || (data[2].qvalue <= 150))) {
    if (LIKELY(false || (data[2].qvalue <= 86))) {
      result[0] += -1.559693370405162e-05;
    } else {
      result[0] += 8.018493624717838e-05;
    }
  } else {
    result[0] += 0.0007062082614175126;
  }
  if (LIKELY(false || (data[2].qvalue <= 148))) {
    result[0] += -3.500609763214595e-06;
  } else {
    result[0] += 0.0005238382736763112;
  }
  if (LIKELY(false || (data[2].qvalue <= 138))) {
    result[0] += -7.872314087503873e-06;
  } else {
    if (LIKELY(false || (data[1].qvalue <= 40))) {
      result[0] += 0.00016982337133831138;
    } else {
      result[0] += 0.0015582625369628602;
    }
  }
  if (LIKELY(false || (data[2].qvalue <= 152))) {
    result[0] += -1.7980922930840279e-06;
  } else {
    result[0] += 0.0008003180445416104;
  }
  if (LIKELY(false || (data[2].qvalue <= 138))) {
    result[0] += -6.905273724778028e-06;
  } else {
    result[0] += 0.00018194518281833516;
  }
  if (LIKELY(false || (data[1].qvalue <= 48))) {
    result[0] += -7.5322283086623e-06;
  } else {
    result[0] += 0.00013574551159436845;
  }
  if (LIKELY(false || (data[2].qvalue <= 152))) {
    result[0] += -1.5802465245244315e-06;
  } else {
    result[0] += 0.0007009422698652187;
  }
  if (LIKELY(false || (data[2].qvalue <= 142))) {
    result[0] += -3.6413942554064128e-06;
  } else {
    result[0] += 0.0002746418585991201;
  }
  if (UNLIKELY(false || (data[2].qvalue <= 12))) {
    result[0] += -3.379293117509959e-05;
  } else {
    result[0] += 2.828160884444577e-05;
  }

  // Apply base_scores
  result[0] += 0;

  // Apply postprocessor
  if (!pred_margin) { postprocess(result); }
}

void bounds_strengthening_predictor::postprocess(double* result)
{
  // Do nothing
}

// Feature names array
const char*
  bounds_strengthening_predictor::feature_names[bounds_strengthening_predictor::NUM_FEATURES] = {
    "m", "n", "nnz_processed", "nnz", "bounds_changed"};
