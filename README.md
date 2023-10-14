# Address Generation Tool with AI-driven Typo Injection 🏡

This repository aims to generate addresses with a structural understanding of each component. It's ideal for testing address-related applications with diverse and real-life-like datasets, especially since it can also introduce typos! 🧪📝

## Current Progress 📈

The primary components for address generation are only being implemented; Level generation is completed, however, flat, house number, street, and general details are still to be done. The typo injection module has been integrated and is currently operational.

## Structure of the Address 📄

The addresses have various components like:
- Level (e.g., `the 1st floor`)
- Flat (e.g., `1C apartment`)
- House number (e.g., `35D-37D`)
- Street (e.g., `Abbey North Road`)
- County (e.g., `Essex`)
- Country (e.g., `England`)
- Postcode (e.g., `6XB IOP`)

Each part has detailed logic to mimic real-world complexity and variations.

## Key Features 🌟

1. **Level Variation**: Optional inclusion with flexible ordering.
2. **Flat Numbering**: Flat numbers can optionally have suffixes.
3. **House Numbering**: Optional inclusion with the capability for ranged house numbers (e.g., 35-37).
4. **Street**: Always included with optional suffix and type.
5. **General**: Contains county, state, and postcode with their respective probabilities.

The addresses can be joined in various combinations to mimic the randomness found in real-world data.

## Artificial Typos 😜

The tool uses a method called `generate_typo` to introduce artificial typos, making the dataset even more challenging and reflective of real-world scenarios.

## Usage 💼

0. Install modules listed in `requirements.txt`
1. Configure probabilities in the `config` module to guide the address generation process.
2. Call the `generate()` function.
3. Get diverse address data!

## Modules 📦

- `lookup`: Contains lists and lookup functions to pull data like street names, flat types, etc.
- `typo`: Houses the function to introduce artificial typos.
- `config`: Contains configuration probabilities and parameters.

## Developer's Guide 🛠

The core of this tool is the `generate()` function. It uses helper functions like `generate_level()`, `generate_flat()`, and so on for each address part. These functions utilize the configurations from the `config` to generate addresses.

You can modify the `lookup` and `typo` modules to adjust the source data and typo logic respectively.

## Understanding `join_str_and_labels` 🧩

One of the foundational utilities in this address generation toolkit is the `join_str_and_labels` function. This function seamlessly integrates multiple parts of the address, ensuring they are correctly spaced and labeled. Let's delve deeper into its mechanics:

### Purpose 🎯

Addresses are comprised of various components (level, flat, house number, etc.). Each of these parts can be represented as a tuple of a string (the address part itself) and a label matrix that tags each character with its type (e.g., street, flat number, level type, etc.). The role of `join_str_and_labels` is to concatenate these tuples into a complete address string and a full label matrix.

### How it Works ⚙️

1. **Filtering Empty Parts**: The function starts by filtering out any parts of the address that don't have content. This ensures that we're only working with meaningful pieces of data.

2. **Generating Separators**: Before joining address parts, we often need separators (like commas or spaces). The function supports using a static separator (like a single space) or dynamic ones that can vary between joins. Dynamic separators are generated using a function passed as the `sep` parameter.

3. **String Concatenation**: The individual strings from the parts are joined together using the predetermined separators. This results in a cohesive address string.

4. **Label Matrix Joining**: Alongside the string concatenation, the label matrices of the parts are also integrated. They're combined using a similar logic to the strings, ensuring that each character in the final address string has a corresponding label in the label matrix.

5. **Output**: The function outputs a tuple of the final address string and its corresponding label matrix. This data can then be used for further processing or testing.

### Why It's Crucial 🌉

Joining address components might seem straightforward, but when we're considering the variability in real-world data, ensuring consistent separators, and managing labeling, things get intricate. `join_str_and_labels` bridges the gap between individual address components and the formation of structured, labeled address data. It's the linchpin that holds the address generation logic together.


## References
<ul>
<li>
<a href="https://towardsdatascience.com/addressnet-how-to-build-a-robust-street-address-parser-using-a-recurrent-neural-network-518d97b9aebd">Address Parser for Australia</a>
</li>
<li>
<a href="https://www.geonames.org/">GeoNames</a> (Postcode, County and Country database for UK)
</li>
<li>
<a href="https://www.streetlist.co.uk">Street List for UK</a>
</li>
</ul>