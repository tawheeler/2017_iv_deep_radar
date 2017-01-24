# Deep Stochastic Radar Models

Supplementary material for _Deep Stochastic Radar Models_, by T. Wheeler, M. Holder, H. Winner, and M. Kochenderfer, submitted to IV 2017.

## Model Training Scripts

This repository contains the model training scripts, which define the model architectures and training procedure.
All scripts run in python 2.7 and train models using [Keras](https://github.com/fchollet/keras) with the [Tensorflow](https://www.tensorflow.org/) backend.

The scripts use the argparse package to allow for argument passing from the command line.
All scripts support:

| parameter  | type  | default | description |
|------------|-------|-------|---------------------------|
| batch_size | int   | 16    | the batch size            |
| nb_epoch   | int   | 100   | number of training epochs |
| verbosity  | int   | 1     | whether to be verbose     |
| train      | int   | 1  | whether to train the model |
| l2         | float | 0.001 | l2 weight regularization  |
| act        | str   | relu  | the activation function   |
| opt        | str   | adadelta  | the optimizer         |
| dset       | str   | dset_runway.h5  | path to the .h5 training dataset |
| tbdir      | str   | /tmp/keras_drm_MODELNAME/ | the tensorboard directory, used for log files based on execution time |
| tblog      | str   | '' | one can optionally specify the exact log filepath |
| tblog      | str   | '' | one can optionally specify the exact tensorboard log filepath |
| save       | str   | ''  | optional path to the save file containing the model weights |
| load       | str   | ''  | optional path to a file containing the model weights to load before training |
| save_pred  | str   | ''  | where to save model predictions |

Some parameters exist for specific models, such as `nrGaussians` for the GMM model.  The GAN model does not provide `nb_epoch`, but `nsteps` instead.

## Data

The training scripts are set up to load data from a `.h5` file. These files should contain at least six entries: `O_train`, `O_test`, `T_train`, `T_test`, `Y_train`, and `Y_test`.
Here, O corresponds to the object list input, `T` corresponds to the terrain input, and `Y` corresponds to the model output - the log power field.

Each entry is a 4-dimensional tensor. The scripts are set up assuming these tensors were generated in (Julia)[http://julialang.org/], which stored tensors in column-major order. Python stores tensors in row-major order.
When saving the entries to the .h5 file in Julia, the dimensions are:

```julia
O = zeros(Float32, N_SAMPLES, MAX_N_OBJECTS, 1, N_OBJECT_FEATURES)
T = zeros(Float32, N_SAMPLES, 64, 64, 1)
Y = zeros(Float32, N_SAMPLES, 64, 64, 1)
```

Both the radar grid and the terrain grid were 64 by 64. These are not requirements.
The maximum number of objects in our experiments was 4.
The number of object features was 10:

feature | description |
|-------|-------------|
| 1     | the object relative x position |
| 2     | the object relative y position |
| 3     | the object relative orientation |
| 4     | relative x-velocity |
| 5     | relative y-velocity |
| 6     | one-hot encoding indicator for CCR |
| 7     | one-hot encoding indicator for VW Golf |
| 8     | one-hot encoding indicator for Yellow target |
| 9     | one-hot encoding indicator for Black target |
| 10    | indicator for an unnused row (no object) |

The object list was standardized over each feature. Neither the terrain input (which was a polar grid of indicator values determining whether a radar grid cell was occupied or not) nor the radar output (which was a polar grid of log power values) were standardized.

## Radar Polar Grid

The radar polar grid had a range from 0.5 to 75 meters and azimuths from -45 to 45 degrees, with 64 evenly-spaced bins in each dimension.

Radar point measurements with a given location and log power were rendered to the radar grid using 2x2 supersampling on the radar grid:

```julia
typealias RadarGrid Array{Float32, 3} # [range, azimuth, layer]
type RadarGridDef
    ranges::LinearDiscretizer
    azimuths::LinearDiscretizer
end
type SupersamplingRandom
    n_samples::Int
end
function fill_cell_max!(
    grid::RadarGrid,
    bin_r::Int,
    bin_a::Int,
    layer::Int,
    fill_value::Real,
    )

    grid[bin_r, bin_a, layer] = max(grid[bin_r, bin_a, layer], fill_value)
end
function render!(supersampling_method::SupersamplingGrid,
    grid::RadarGrid,
    def::RadarGridDef,
    geom::Geom,
    bin_r::Int,
    bin_a::Int,
    layer::Int,
    fill_value::Real,
    )

    n_hits = 0

    a_lo, a_hi = def.azimuths.binedges[bin_a], def.azimuths.binedges[bin_a+1]
    a_Δ = (a_hi - a_lo)/supersampling_method.n_azimuth

    r_lo, r_hi = def.ranges.binedges[bin_r], def.ranges.binedges[bin_r+1]
    r_Δ = (r_hi - r_lo)/supersampling_method.n_range

    a, a_i = a_lo + a_Δ/2, 1
    while a_i ≤ supersampling_method.n_azimuth

        cosa, sina = cos(a), sin(a)

        r, r_i = r_lo + r_Δ/2, 1
        while r_i ≤ supersampling_method.n_range

            P = VecE2(r*cosa, r*sina)
            n_hits += contains(geom, P)

            r_i += 1
            r += r_Δ
        end

        a_i += 1
        a += a_Δ
    end

    # fill with the fractional number of hits, as appropriate
    fill_cell_max!(grid, bin_r, bin_a, layer,
                   fill_value * n_hits/(supersampling_method.n_azimuth * supersampling_method.n_range))

    grid
end

POWER_LOG_OFFSET = 100.0f0
OBJ_RADIUS = 2.0
SUPERSAMPLE = SupersamplingGrid(2,2)

function _set_Y!(
    Y::Array{Float32, 4},
    batch_index::Int,
    clusters::Clusters,
    )

    layer_index = 1
    clear_layer!(radargrid, layer_index)
    for obj in clusters
        geom = Circ(get_pos(obj), OBJ_RADIUS)
        render!(radargrid, radargrid_def, geom, layer_index, obj.power_log + POWER_LOG_OFFSET, supersampling_method)
    end
    Y[batch_index, :, :, 1] = RG_Y[:, :, layer_index]

    Y
end
```

## Contact

Please feel free to file an issue if you have any questions.
