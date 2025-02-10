# -*- coding: utf-8 -*-
''' Fraction To Text

Description:
Takes a fraction (ie, 1/2) and outputs english words (ie, one-half)

@author zhou <zyzhou@stu.xmu.edu.cn>
@date 08/18 2019

'''

import re

scale = [
    '', 'thousand', 'million', 'billion', 'trillion', 'quadrillion',
    'quintillion', 'sextillion', 'octillion', 'nonillion', 'decillion',
    'undecillion', 'duodecillion', 'tredecillion', 'quattuordecillion',
    'quindecillion', 'sexdecillion', 'septendecillion', 'octodecillion',
    'noverndecillion', 'vigintillion'
]

cardinals = [
    '', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
    'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen', 'fourteen',
    'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen',
]

tensCardinals = [
    '', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy',
    'eighty', 'ninety',
]

ordinals = [
    '', 'first', 'second', 'third', 'fourth', 'fifth', 'sixth',
    'seventh', 'eighth', 'ninth', 'tenth', 'eleventh', 'twelfth',
    'thirteenth', 'fourteenth', 'fifteenth', 'sixteenth', 'seventeenth',
    'eighteenth', 'nineteenth'
]

tensOrdinals = [
    '', '', 'twentieth', 'thirtieth', 'fortieth', 'fiftieth', 'sixtieth',
    'seventieth', 'eightieth', 'ninetieth',
]


class FractionToText:

    def __init__(self):

        self._integer = False
        self._fraction = False
        self._numerator = False
        self._denominator = False

        self._scale = scale
        self._cardinals = cardinals
        self._tensCardinals = tensCardinals
        self._ordinals = ordinals
        self._tensOrdinals = tensOrdinals

    def convert(self, fraction):
        # clean up
        fraction = fraction.strip()
        fraction = re.sub(r'\s+', ' ', fraction)

        # get parts
        parts = fraction.split()

        if '/' not in fraction or len(parts) > 2:
            return fraction
        elif len(parts) == 2:
            self._integer = parts[0]
            self._fraction = parts[1]
        else:
            self._fraction = parts[0]

        (self._numerator, self._denominator) = self._fraction.split('/')

        # covert number to text
        integer = self.number_to_cardinal(self._integer)
        numerator = self.number_to_cardinal(self._numerator)

        if int(self._denominator) == 2:
            denominator = 'halves' if int(self._numerator) > 1 else 'half'
        elif int(self._denominator) == 4:
            denominator = 'quarters' if int(self._numerator) > 1 else 'quarter'
        else:
            denominator = self.number_to_ordinal(self._denominator)

        glue = '-' if '-' not in denominator else ' '
        fraction = numerator + glue + denominator
        fraction = re.sub(r'\s*-\s*', '-', fraction)

        if int(self._numerator) > 1 and int(self._denominator) not in [2, 4]:
            fraction = fraction + 's'

        # whole fraction
        if integer:
            fraction = integer + ' and ' + fraction

        return fraction.strip()

    def number_to_list(self, number):
        return re.findall(r'.{3}', number.zfill(len(number) // 3 * 3 + 3))

    def number_to_english(self, number):
        hundreds = int(number) // 100
        tens = int(number) % 100
        pre = self._cardinals[hundreds]+' hundred' if hundreds else ''

        if tens < 20:
            post = self._cardinals[tens]
        else:
            post = self._tensCardinals[tens // 10]

            if not tens % 10 == 0:
                post = post + '-' + self._cardinals[tens % 10]

        if pre and post:
            return (pre + ' ' + post).strip()

        return (pre + post).strip()

    def number_to_cardinal(self, number):
        if not number:
            return number

        number_list = self.number_to_list(number)[::-1]
        cardinal_list = []
        for i, number in enumerate(number_list):
            english_number = self.number_to_english(number)

            if english_number:
                cardinal_list.append(english_number + ' ' + self._scale[i])

        post = cardinal_list[0]
        pre = ' '.join(cardinal_list[1:][::-1])

        if pre and post:
            return (pre + ' ' + post).strip()

        return (pre + post).strip()

    def cardinal_to_ordinal(self, cardinal):
        words = cardinal.split()
        post = words[-1]

        if '-' in post:
            parts = post.split('-')
            pre = parts[0]
            post = parts[1]

        if post in self._cardinals:
            post = self._ordinals[self._cardinals.index(post)]
        elif post in self._tensCardinals:
            post = self._tensOrdinals[self._tensCardinals.index(post)]
        else:
            post = post + 'th'

        if 'pre' in vars():
            post = pre + '-' + post

        words[-1] = post

        if len(words) > 1:
            if '-' not in post:
                post = words.pop()
                pre = words.pop()
                last = pre + '-' + post
                words.append(last)

        return ' '.join(words).strip()

    def number_to_ordinal(self, number):
        return self.cardinal_to_ordinal(self.number_to_cardinal(number)).strip()
