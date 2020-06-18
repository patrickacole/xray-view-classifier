from enum import IntEnum

class Labels(IntEnum):
    # Labels present in the CheXpert csv files
    LATERAL   = 0
    FRONTAL   = 1

    def convert(s):
        """
        Convert csv cell to one of the labels

        @param s : csv cell data
        @return label : enum label for that cell
        """
        if s == 'Lateral': return Labels.LATERAL
        else: return Labels.FRONTAL
