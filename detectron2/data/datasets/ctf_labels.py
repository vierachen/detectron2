# author: chenwr 
# datetime: 20191210
# function: ctf label information

from collections import namedtuple
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name        id      trainId     category    catId    color
    Label(  'ctf'   ,   0 ,      255 ,      'void',     0 ,     (  255,  0,  0) ),
    Label(  'gold'  ,   1 ,      255 ,      'void',     0 ,     (  0,  255,  0) ),
    Label(  '999'   ,   2 ,      255 ,      'void',     0 ,     (  0,  0,  255) ),
    Label(  'other' ,   3 ,      255 ,      'void',     0 ,     (  0,  0,  255) ),
]


labels_ctf = [
    #       name        id      trainId     category    catId    color
    Label(  'c'     ,   0 ,      255 ,      'void',     0 ,     (  255,  0,  0) ),
    Label(  't'     ,   1 ,      255 ,      'void',     0 ,     (  0,  255,  0) ),
    Label(  'f'     ,   2 ,      255 ,      'void',     0 ,     (  0,  0,  255) ),
]

labels_gold = [
    #       name        id      trainId     category    catId    color
    Label(  '足'    ,   0 ,      255 ,      'void',     0 ,     (  255,  0,  0) ),
    Label(  '金'    ,   1 ,      255 ,      'void',     0 ,     (  0,  255,  0) ),
]

labels_999 = [
    #       name        id      trainId     category    catId    color
    Label(  '9'     ,   0 ,      255 ,      'void',     0 ,     (  255,  0,  0) ),
    Label(  '9'     ,   1,      255 ,      'void',     0 ,     (  0,  255,  0) ),
    Label(  '9'     ,   2,      255 ,      'void',     0 ,     (  0,  0,  255) ),
]
