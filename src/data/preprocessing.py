import numpy as np


face_landmarks = list(
    range(len([0, 9, 11, 13, 14, 17, 117, 118, 119, 199, 346, 347, 348]))
)
left_hand_landmarks = list(range(len(face_landmarks), len(face_landmarks) + 21))
pose_landmarks = list(
    range(
        len(face_landmarks) + len(left_hand_landmarks),
        len(face_landmarks) + len(left_hand_landmarks) + 33,
    )
)
right_hand_landmarks = list(
    range(
        len(face_landmarks) + len(left_hand_landmarks) + len(pose_landmarks),
        len(face_landmarks) + len(left_hand_landmarks) + len(pose_landmarks) + 21,
    )
)

landmark_idx = [0, 9, 11, 13, 14, 17, 117, 118, 119, 199, 346, 347, 348] + list(
    range(468, 543)
)


def normalize(data):
    # Face
    data[:, face_landmarks] = data[:, face_landmarks] - data[:, face_landmarks].mean(
        axis=0
    )
    data[:, face_landmarks] = data[:, face_landmarks] / (
        data[:, face_landmarks].std(axis=0) + 1e-5
    )

    # Left hand
    data[:, left_hand_landmarks] = data[:, left_hand_landmarks] - data[
        :, left_hand_landmarks
    ].mean(axis=0)
    data[:, left_hand_landmarks] = data[:, left_hand_landmarks] / (
        data[:, left_hand_landmarks].std(axis=0) + 1e-5
    )

    # Pose
    data[:, pose_landmarks] = data[:, pose_landmarks] - data[:, pose_landmarks].mean(
        axis=0
    )
    data[:, pose_landmarks] = data[:, pose_landmarks] / (
        data[:, pose_landmarks].std(axis=0) + 1e-5
    )

    # Right hand
    data[:, right_hand_landmarks] = data[:, right_hand_landmarks] - data[
        :, right_hand_landmarks
    ].mean(axis=0)
    data[:, right_hand_landmarks] = data[:, right_hand_landmarks] / (
        data[:, right_hand_landmarks].std(axis=0) + 1e-5
    )
    return data


def preprocess_data(
    x: np.ndarray, max_sequence_length: int, do_normalize: bool, do_substract: bool
) -> np.ndarray:
    x = resize_timeseries(x, max_sequence_length)
    x = np.nan_to_num(x)
    x = keep_only_relevant_data(x)

    if do_substract:
        x = substract_distance(x)
    if do_normalize:
        x = normalize(x)
    x = x.reshape(x.shape[0], -1)
    # convert from [window_size, features] to [features, window_size]
    x = x.T
    return x


def resize_timeseries(data: np.ndarray, new_shape) -> np.ndarray:
    old_shape = data.shape[0]
    if old_shape == new_shape:
        return data
    else:
        # If the timeseries is longer than new_shape, resample by selecting every other point
        if old_shape > new_shape:
            indices = np.arange(0, old_shape, old_shape // new_shape)[:new_shape]
            return data[indices]
        # If the timeseries is shorter than new_shape, pad with zeros to get new_shape points
        else:
            new_x = np.zeros((new_shape, *data.shape[1:]), dtype=data.dtype)
            new_x[: min(data.shape[0], new_shape)] = data[:new_shape]
            return new_x


def keep_only_relevant_data(data: np.ndarray) -> np.ndarray:
    return data[:, landmark_idx]


# def substract_distance(data: np.ndarray) -> np.ndarray:
#     # Face
#     data[:, face_landmarks[1:]] = (
#         data[:, face_landmarks[1:]] - data[:, face_landmarks[0:1]]
#     )

#     # Left hand
#     data[:, left_hand_landmarks[1:]] = (
#         data[:, left_hand_landmarks[1:]] - data[:, left_hand_landmarks[0:1]]
#     )

#     # Pose
#     data[:, pose_landmarks[1:]] = (
#         data[:, pose_landmarks[1:]] - data[:, pose_landmarks[0:1]]
#     )

#     # Right hand
#     data[:, right_hand_landmarks[1:]] = (
#         data[:, right_hand_landmarks[1:]] - data[:, right_hand_landmarks[0:1]]
#     )
#     return data


def substract_distance(data: np.ndarray) -> np.ndarray:
    data = calculate_distance_hand(data, left_hand_landmarks)
    data = calculate_distance_hand(data, right_hand_landmarks)
    return data


def calculate_distance_hand(data, hand_landmarks):
    thumb_index = data[:, hand_landmarks[4]] - data[:, hand_landmarks[8]]
    thumb_middle = data[:, hand_landmarks[4]] - data[:, hand_landmarks[12]]
    thumb_ring = data[:, hand_landmarks[4]] - data[:, hand_landmarks[16]]
    thumb_pinky = data[:, hand_landmarks[4]] - data[:, hand_landmarks[20]]

    index_middle = data[:, hand_landmarks[8]] - data[:, hand_landmarks[12]]
    middle_ring = data[:, hand_landmarks[12]] - data[:, hand_landmarks[16]]
    ring_pinky = data[:, hand_landmarks[16]] - data[:, hand_landmarks[20]]

    # thumb_index is of shape (window_size, 3) but data is of shape (window_size, 21, 3), concatenate along axis 1
    data = np.concatenate(
        (
            data,
            thumb_index[:, None],
            thumb_middle[:, None],
            thumb_ring[:, None],
            thumb_pinky[:, None],
            index_middle[:, None],
            middle_ring[:, None],
            ring_pinky[:, None],
        ),
        axis=1,
    )
    return data
