import numpy as np
import re


def parse_reaction(reaction, y, dy_idx, params, dys):
    """ Parse the reaction and update the lists of species, parameters and the derivative of the species.
    
    Parameters
    ----------
    reaction : str
        The reaction in the form of 'aA + bB <-> cC + dD, kf, kr'.
    y : list of str
        List of species.
    dy_idx : dict of index
        Dict of the derivative of the species (name: i).
    params : list of str
        List of parameters in the reaction rates.
    dys : list of str
        List of the derivative of the species.

    Returns
    -------
    None
    """
    parts = re.split(r'<?-?>|,', reaction)
    # print(parts)
    reactants_str = parts[0].strip()
    products_str = parts[1].strip()
    kf_str = parts[2].strip()
    kr_str = parts[3].strip() if len(parts) > 3 else '0'
    # print(reactants_str, products_str, kf_str, kr_str)
    
    # Parse reactants (only positive coefficients expected)
    reactants = re.findall(r'(\d*)\s*(\w+)', reactants_str)
    
    # Parse products (handle both positive and negative coefficients)
    # Split by + and - while keeping the signs
    product_parts = re.split(r'(\+|\-)', products_str)
    products = []
    
    i = 0
    while i < len(product_parts):
        part = product_parts[i].strip()
        if part == '+':
            i += 1
            if i < len(product_parts):
                next_part = product_parts[i].strip()
                match = re.match(r'(\d*)\s*(\w+)', next_part)
                if match:
                    coeff, species = match.groups()
                    products.append((coeff, species))
        elif part == '-':
            i += 1
            if i < len(product_parts):
                next_part = product_parts[i].strip()
                match = re.match(r'(\d*)\s*(\w+)', next_part)
                if match:
                    coeff, species = match.groups()
                    # Make coefficient negative
                    if coeff == '':
                        coeff = '-1'
                    else:
                        coeff = '-' + coeff
                    products.append((coeff, species))
        elif part:  # First term (no sign means positive)
            match = re.match(r'(\d*)\s*(\w+)', part)
            if match:
                coeff, species = match.groups()
                products.append((coeff, species))
        i += 1
    
    # print(reactants, products)

    # parse the parameters from kf_str and kr_str
    # it can be number*parameter*parameter or parameter*parameter or parameter
    # the parameters are separated by '*'; the number is optional
    def parse_rate_params(rate_str):
        rate_parts = re.findall(r'(\d*\.?\d*)\s*\*?\s*(\w+)', rate_str)
        # print(rate_parts)
        rate_params = []
        for coeff, param in rate_parts:
            if param not in params:
                n = len(params)
                params[param] = f'params[{n}]'
            rate_params.append(f'{coeff}*{params[param]}' if coeff else params[param])
        # print('rate_params: ', rate_params)
        return None
        

    # Update the list of parameters
    parse_rate_params(kf_str)
    if kr_str != '0':
        parse_rate_params(kr_str)

    # Update the list of species
    for _, species in reactants + products:
        if species not in y:
            n = len(y)
            y[species] = f'y[{n}]'
            dy_idx[species] = n

    # Update the list of the derivative of the species
    # reactants
    # print("reactants: ", reactants)
    left =''
    for n, species in reactants:
        if n:
            if left:
                left += '*'
            left += f'{y[species]}**{n}'
        else:
            if left:
                left += '*'
            left += f'{y[species]}'
    
    # print("products: ", products)
    right = ''
    for n, species in products:
        # Only use positive coefficients for the right side rate expression
        abs_n = n.lstrip('-') if n.startswith('-') else n
        if abs_n:
            if right:
                right += '*'
            right += f'{y[species]}**{abs_n}'
        else:
            if right:
                right += '*'
            right += f'{y[species]}'

    # Handle reactants (consumption terms)
    for n, species in reactants:
        idx = dy_idx[species]
        if len(dys) <= idx:
            dys.extend([None] * (idx - len(dys) + 1))
        if n == '':
            if kr_str == '0':
                if dys[idx] is None:
                    dys[idx] = f'-{kf_str}*{left}'
                else:
                    dys[idx] += f'-{kf_str}*{left}'
            else:
                if dys[idx] is None:
                    dys[idx] = f'-{kf_str}*{left}+{kr_str}*{right}'
                else:
                    dys[idx] += f'-{kf_str}*{left}+{kr_str}*{right}'
        else:
            if kr_str == '0':
                if dys[idx] is None:
                    dys[idx] = f'-{kf_str}*{n}*{left}'
                else:
                    dys[idx] += f'-{kf_str}*{n}*{left}'
            else:
                if dys[idx] is None:
                    dys[idx] = f'-{kf_str}*{n}*{left}+{kr_str}*{n}*{right}'
                else:
                    dys[idx] += f'-{kf_str}*{n}*{left}+{kr_str}*{n}*{right}'

    # Handle products (production/consumption terms based on sign)
    for n, species in products:
        idx = dy_idx[species]
        if len(dys) <= idx:
            dys.extend([None] * (idx - len(dys) + 1))
        
        # Determine if this is a consumption (negative) or production (positive) term
        is_negative = n.startswith('-')
        abs_n = n.lstrip('-') if is_negative else n
        
        if abs_n == '':
            abs_n = '1'
        
        if is_negative:
            # This species is consumed on the product side
            if kr_str == '0':
                if dys[idx] is None:
                    dys[idx] = f'-{kf_str}*{abs_n}*{left}'
                else:
                    dys[idx] += f'-{kf_str}*{abs_n}*{left}'
            else:
                if dys[idx] is None:
                    dys[idx] = f'-{kf_str}*{abs_n}*{left}+{kr_str}*{abs_n}*{right}'
                else:
                    dys[idx] += f'-{kf_str}*{abs_n}*{left}+{kr_str}*{abs_n}*{right}'
        else:
            # This species is produced on the product side
            if abs_n == '1':
                if kr_str == '0':
                    if dys[idx] is None:
                        dys[idx] = f'+{kf_str}*{left}'
                    else:
                        dys[idx] += f'+{kf_str}*{left}'
                else:
                    if dys[idx] is None:
                        dys[idx] = f'+{kf_str}*{left}-{kr_str}*{right}'
                    else:
                        dys[idx] += f'+{kf_str}*{left}-{kr_str}*{right}'
            else:
                if kr_str == '0':
                    if dys[idx] is None:
                        dys[idx] = f'+{kf_str}*{abs_n}*{left}'
                    else:
                        dys[idx] += f'+{kf_str}*{abs_n}*{left}'
                else:
                    if dys[idx] is None:
                        dys[idx] = f'+{kf_str}*{abs_n}*{left}-{kr_str}*{abs_n}*{right}'
                    else:
                        dys[idx] += f'+{kf_str}*{abs_n}*{left}-{kr_str}*{abs_n}*{right}'

    # print(y)
    # print(params)
    # print(dys)


def generate_ode_system(reactions):
    """ Generate the ODE system for the given reactions.

    Parameters
    ----------
    reactions : list of str
        List of str, each str represents a reaction. The reaction should be in the form of
        'aA + bB <-> cC + dD, kf, kr', where a, b, c, d are integers and A, B, C, D are species.
        if there is no integer, it is assumed to be 1. kf is the forward rate constant and kr is the
        reverse rate constant. the number of reactants and products should are not limited. the species name
        can be more than one character. if the reaction is irreversible, the reverse rate constant should be 0.
        the reaction rates can be expressed by mathematical expression. for example, kf can be written as 2 * gamma * kf1,
        where gamma is a input parameter and kf1  is a input paramter.

    Returns
    -------
    str
        The content of a function that takes the current state of the system and returns the derivative of the state.
    """
    y = {} # dict of species (name: y[i]), including reactants and products
    dy_idx = {} # dict of the derivative of the species (name: i)
    params = {} # dict of parameters (name: params[i]) in the reaction rates
    dys = [] # list of the derivative of the species

    for reaction in reactions:
        parse_reaction(reaction, y, dy_idx, params, dys)

    # write the content of the function
    output = 'def ode(self, y, t, params):\n'
    for key, value in y.items():
        output += f'\t# {value} = {key}\n'
    for p in params:
        output += f'\t{p} = {params[p]}\n'

    output += f'\tdylist = []\n'
    for i, dy in enumerate(dys):
        if dy is None:
            output += f'\tdylist.append(0)\n'
        else:
            output += f'\tdylist.append({dy})\n'
    output += f'\treturn np.array(dylist)\n'

    print(output)
    return output